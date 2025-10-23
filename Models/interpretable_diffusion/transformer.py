import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, Transpose, GELU2, series_decomp


class TrendBlock(nn.Module):
    """
    使用多项式基回归拟合时间序列的趋势项。

    参数：
    - in_dim(int): 输入通道数（通常对应时间步 T，作为 Conv1d 的 in_channels）。
    - out_dim(int): 输出通道数（通常与 in_dim 相同）。
    - in_feat(int): 输入特征维度（embedding 维）。
    - out_feat(int): 输出特征维度。
    - act(nn.Module): 激活函数。
    - trend_poly(int): 多项式阶数（默认3）。经验值，足以拟合平滑趋势，过高易过拟合。
    """
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act, trend_poly: int = 3):
        super(TrendBlock, self).__init__()
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )

        # 使用 float64，前向时再对齐到输入 x 的 dtype/device
        lin_space = torch.arange(1, out_dim + 1, 1, dtype=torch.float64) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        """
        输入/输出形状：
        - input: (B, C, H) 其中 C=时间步数 T，H=embedding 维
        - 返回: (B, C, H)
        """
        b, c, h = input.shape  # (B, C=T, H)
        x = self.trend(input).transpose(1, 2)
        # 将多项式基对齐到输入的 dtype/device
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device, dtype=x.dtype))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals
    

class MovingBlock(nn.Module):
    """
    Model trend of time series using the moving average.
    """
    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return x, trend_vals


class FourierLayer(nn.Module):
    """
    使用逆 DFT 思想建模时间序列的季节性（选取能量最高的若干频率分量）。

    参数：
    - d_model(int): embedding 维度。
    - low_freq(int): 忽略最低的若干频率（默认1），用于去除直流与超低频噪声的经验值。
    - factor(float): 选频比例系数，top_k = max(1, int(factor * log(T)))。
    - topk_min(int): 选频的最小个数下限（默认1）。
    """
    def __init__(self, d_model, low_freq=1, factor=1, topk_min: int = 1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq
        self.topk_min = topk_min

    def forward(self, x):
        """
        x: (B, T, D)
        返回: (B, T, D)
        """
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        # 频率张量对齐到输入的 dtype/device
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device, dtype=x_freq.real.dtype)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        # 使用与 x_freq.real 相同的 dtype/device 生成时间索引
        t = rearrange(torch.arange(t, dtype=x_freq.real.dtype, device=x_freq.device),
                      't -> () () t ()')

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        """
        选取能量最高的 top-k 个频率分量。
        - x_freq: (B, F, D) 复数张量
        - 返回: (x_freq_selected, index_tuple)
        说明：top_k = max(topk_min, int(factor * log(max(length, 2))))
        """
        length = x_freq.shape[1]
        top_k = max(self.topk_min, int(self.factor * math.log(max(length, 2))))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple
    

class SeasonBlock(nn.Module):
    """
    使用有限项傅里叶级数拟合季节性。

    参数：
    - in_dim(int): 输入通道数（时间步 T）。
    - out_dim(int): 输出通道数。
    - factor(int): 每个基的倍数因子，控制多项式/谐波数量。
    - season_max_harmonics(int): 谐波数上限（默认32）。经验值，避免过多谐波导致过拟合。
    """
    def __init__(self, in_dim, out_dim, factor=1, season_max_harmonics: int = 32):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(season_max_harmonics, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1, dtype=torch.float64) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 \
            else (season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device, dtype=x.dtype))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


class FullAttention(nn.Module):
    def __init__(self,
                 n_embd,          # embedding dimension
                 n_head,          # number of attention heads
                 attn_pdrop=0.1,  # attention dropout
                 resid_pdrop=0.1, # residual dropout
                 use_rope=False,  # 🔹是否启用RoPE位置编码
                 max_seq_len=512  # 🔹RoPE最大序列长度（可调）
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.use_rope = use_rope

        # QKV投影
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # 正则化与输出
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)

        # 🔹若启用RoPE，初始化旋转位置编码
        if self.use_rope:
            head_dim = n_embd // n_head
            self.rotary_emb = RotaryEmbedding(dim=head_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor (B, T, C)
            mask: Optional attention mask
        """
        B, T, C = x.size()

        # --- Q, K, V计算 ---
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)    # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # --- 🔹RoPE旋转位置编码（仅当启用时） ---
        if self.use_rope:
            pos = torch.arange(T, device=x.device)
            rotary_pos_emb = self.rotary_emb(pos).to(x.dtype)
            q, k = apply_rotary_emb(rotary_pos_emb, q, k)

        # --- 标准点乘注意力 ---
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # (B, nh, T, T)
        if mask is not None:
            # 自动调整mask形状
            # att.shape: (B, n_head, T, T)
            # mask可能是(B, T)或(B, 1, 1, T)
            if mask.dim() == 2:
                # 扩展到(B, 1, 1, T)
                mask = mask[:, None, None, :]
            elif mask.dim() == 3 and mask.shape[1] == 1:
                # 扩展到(B, 1, T, T)
                mask = mask[:, :, None, :]
            # 自动裁剪或pad到当前注意力长度
            T_att = att.size(-1)
            T_mask = mask.size(-1)
            if T_mask < T_att:
                # 若mask短于序列，右侧pad
                pad_len = T_att - T_mask
                mask = F.pad(mask, (0, pad_len), value=1)
            elif T_mask > T_att:
                # 若mask长于序列，裁剪
                mask = mask[..., :T_att]
            att = att.masked_fill(mask == 0, torch.finfo(att.dtype).min)


        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # --- 聚合输出 ---
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.resid_drop(self.proj(y))

        # 返回平均注意力图（用于可视化等）
        att_mean = att.mean(dim=1, keepdim=False)
        return y, att_mean


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att
    

class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU'
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
        
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )
        
    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))   # only one really use encoder_output
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.,
        resid_pdrop=0.,
        mlp_hidden_times=4,
        block_activate='GELU',
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
        ) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x


class DecoderBlock(nn.Module):
    """ an unassuming Transformer block (modified with learnable trend/season scaling) """
    def __init__(self,
                 n_channel,
                 n_feat,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 condition_dim=1024,
                 ):
        super().__init__()
        
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop,
                )
        self.attn2 = CrossAttention(
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                )
        
        self.ln1_1 = AdaLayerNorm(n_embd)

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        # 原有趋势与季节模块
        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act)
        self.seasonal = FourierLayer(d_model=n_embd)

        # ✅ 新增：可学习缩放系数
        self.trend_scale = nn.Parameter(torch.tensor(0.0))
        self.season_scale = nn.Parameter(torch.tensor(0.0))

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        # 自注意力
        a, att = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        # 交叉注意力
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a

        # 趋势 + 季节性建模（带缩放系数）
        x1, x2 = self.proj(x).chunk(2, dim=1)
        trend = self.trend_scale * self.trend(x1)
        season = self.season_scale * self.seasonal(x2)

        # MLP 残差块
        x = x + self.mlp(self.ln2(x))

        # 去均值操作
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season

    

class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_feat,
        n_embd=1024,
        n_head=16,
        n_layer=10,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        condition_dim=512    
    ):
      super().__init__()
      self.d_model = n_embd
      self.n_feat = n_feat
      self.blocks = nn.Sequential(*[DecoderBlock(
                n_feat=n_feat,
                n_channel=n_channel,
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                condition_dim=condition_dim,
        ) for _ in range(n_layer)])
      
    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        b, c, _ = x.shape
        # att_weights = []
        mean = []
        season = torch.zeros((b, c, self.d_model), device=x.device, dtype=x.dtype)
        trend = torch.zeros((b, c, self.n_feat), device=x.device, dtype=x.dtype)
        for block_idx in range(len(self.blocks)):
            x, residual_mean, residual_trend, residual_season = \
                self.blocks[block_idx](x, enc, t, mask=padding_masks, label_emb=label_emb)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)
        return x, mean, trend, season


class Transformer(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        n_layer_enc=5,
        n_layer_dec=14,
        n_embd=1024,
        n_heads=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        max_len=2048,
        conv_params=None,
        **kwargs
    ):
        super().__init__()
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)

        if conv_params is None or conv_params[0] is None:
            if n_feat < 32 and n_channel < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        self.combine_s = nn.Conv1d(n_embd, n_feat, kernel_size=kernel_size, stride=1, padding=padding,
                                   padding_mode='circular', bias=False)
        self.combine_m = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0,
                                   padding_mode='circular', bias=False)

        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        self.decoder = Decoder(n_channel, n_feat, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times,
                               block_activate, condition_dim=n_embd)
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

    def forward(self, input, t, padding_masks=None, return_res=False):
        """
        主前向：编码-解码-重构，输出趋势与季节误差。
        - input: (B, T, C)
        - t: (B,)
        - 返回: (trend: (B, T, C), season_error: (B, T, C))
        """
        emb = self.emb(input)
        inp_enc = self.pos_enc(emb)
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks)

        inp_dec = self.pos_dec(emb)
        output, mean, trend, season = self.decoder(inp_dec, t, enc_cond, padding_masks=padding_masks)

        res = self.inverse(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        season_error = self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        trend = self.combine_m(mean) + res_m + trend

        if return_res:
            return trend, self.combine_s(season.transpose(1, 2)).transpose(1, 2), res - res_m

        return trend, season_error


if __name__ == '__main__':
    pass