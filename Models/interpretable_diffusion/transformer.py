import math
import torch
import numpy as np
import torch.nn.functional as F

"""
约定与阅读指南
----------------
本文件实现了用于可解释时序建模的 Transformer 组件，包含趋势/季节性分解与注意力模块。

形状约定（默认）：
- 注意力子层使用序列形状 (B, T, C)，其中 B: batch, T: 序列长度, C: 通道/嵌入维度。
- 卷积与时序分解模块多使用 (B, C, L) 形式，其中 L 通常与 T 对齐；名称差异仅源于历史习惯。
- 在发生 view/transpose/rearrange 的位置，尽量标注英文形状注释便于跟踪维度变化。

说明：以下修改仅增加注释与文档，不改变任何已有逻辑与接口。
"""

from torch import nn
from einops import rearrange, reduce, repeat
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, Transpose, GELU2, series_decomp
from Models.interpretable_diffusion.layers.attention import FullAttention, CrossAttention, FreqAttention
from Models.interpretable_diffusion.layers.decomposition import TrendBlock, MovingBlock, FourierLayer, SeasonBlock


# 注意：FullAttention、CrossAttention、TrendBlock、MovingBlock、FourierLayer、SeasonBlock
# 已迁移至 Models.interpretable_diffusion.layers 子包中；此处仅做导入以保持逻辑不变。
    

class EncoderBlock(nn.Module):
    """
    基础编码器块：AdaLayerNorm + 自注意力 + 前馈（MLP）残差。

    输入/输出：
    - x: (B, T, C) -> (B, T, C)
    - 同步返回平均注意力权重（便于可视化/诊断）。
    """
    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 use_freq_attn=False,
                 freq_size=None,
                 freq_heads=4,
                 freq_head_dim=16,
                 freq_pdrop=0.1,
                 freq_resid_pdrop=0.1,
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

        # optional frequency-axis attention branch（并行分支，门控）
        self.use_freq_attn = use_freq_attn
        if self.use_freq_attn:
            assert freq_size is not None, 'freq_size must be provided when use_freq_attn is True'
            self.freq_attn = FreqAttention(
                n_embd=n_embd,
                freq_size=freq_size,
                n_head=freq_heads,
                head_dim=freq_head_dim,
                attn_pdrop=freq_pdrop,
                resid_pdrop=freq_resid_pdrop,
            )
            # gating parameter for frequency branch (ReZero-like start at 0)
            self.gate_f = nn.Parameter(torch.tensor(0.0))
            self.freq_dropout = nn.Dropout(freq_pdrop)
        
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )
        
    def forward(self, x, timestep, mask=None, label_emb=None):
        # Pre-normalize once and reuse for both time and freq branches
        x_norm = self.ln1(x, timestep, label_emb)
        a, att = self.attn(x_norm, mask=mask)  # a: (B,T,C), att: (B,T,T)
        x = x + a

        # 并行频域分支：以 pre-norm 的输入为基础，计算沿频域的注意力并门控融合
        if getattr(self, 'use_freq_attn', False):
            a_freq, att_freq = self.freq_attn(x_norm)  # a_freq: (B,T,C)
            a_freq = self.freq_dropout(a_freq)
            x = x + self.gate_f * a_freq
        x = x + self.mlp(self.ln2(x))   # MLP 残差
        return x, att


class Encoder(nn.Module):
    """
    编码器：堆叠多个 EncoderBlock，对输入序列进行上下文编码。
    输入/输出： (B, T, C) -> (B, T, C)
    """
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.,
        resid_pdrop=0.,
        mlp_hidden_times=4,
        block_activate='GELU',
        use_freq_attn=False,
        freq_size=None,
        freq_heads=4,
        freq_head_dim=16,
        freq_pdrop=0.1,
        freq_resid_pdrop=0.1,
    ):
        super().__init__()
        self.use_freq_attn = use_freq_attn
        self.blocks = nn.Sequential(*[EncoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
        activate=block_activate,
        use_freq_attn=use_freq_attn,
        freq_size=freq_size,
        freq_heads=freq_heads,
        freq_head_dim=freq_head_dim,
        freq_pdrop=freq_pdrop,
        freq_resid_pdrop=freq_resid_pdrop,
        ) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input  # (B, T, C)
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x


class DecoderBlock(nn.Module):
    """
    解码器块：自注意力 + 交叉注意力 + MLP，并输出趋势/季节性分量。
    同时包含可学习缩放系数（trend_scale/season_scale），便于稳定训练与解释。

    返回：
    - x_minus_mean: (B, T, C)  # 去通道均值后
    - linear_mean: (B, T, n_feat)  # 经线性层映射的均值项
    - trend: (B, C_token, n_feat)
    - season: (B, C_token, d_model)
    """
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
                 use_freq_attn=False,
                 freq_size=None,
                 freq_heads=4,
                 freq_head_dim=16,
                 freq_pdrop=0.1,
                 freq_resid_pdrop=0.1,
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
        # optional frequency attention branch (并联 + 门控)，放在 self-attn 之后、cross-attn 之前
        self.use_freq_attn = use_freq_attn
        if self.use_freq_attn:
            assert freq_size is not None, 'freq_size must be provided when use_freq_attn=True'
            self.freq_attn = FreqAttention(n_embd=n_embd,
                                           freq_size=freq_size,
                                           n_head=freq_heads,
                                           head_dim=freq_head_dim,
                                           attn_pdrop=freq_pdrop,
                                           resid_pdrop=freq_resid_pdrop)
            self.gate_f = nn.Parameter(torch.tensor(0.0))
            self.freq_dropout = nn.Dropout(freq_pdrop)
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
        x_norm = self.ln1(x, timestep, label_emb)
        a, att = self.attn1(x_norm, mask=mask)  # a: (B,T,C)
        x = x + a
        # 并行频域分支（放在自注意力后、交叉注意力前）
        if getattr(self, 'use_freq_attn', False):
            a_freq, att_freq = self.freq_attn(x_norm)  # a_freq: (B,T,C)
            a_freq = self.freq_dropout(a_freq)
            x = x + self.gate_f * a_freq
        # 交叉注意力
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)  # a: (B,T,C)
        x = x + a

        # 趋势 + 季节性建模（带缩放系数）
        x1, x2 = self.proj(x).chunk(2, dim=1)  # (B,2*C,T) -> 两路 (B,C,T)
        trend = self.trend_scale * self.trend(x1)      # (B,C,n_feat) 通过 TrendBlock
        season = self.season_scale * self.seasonal(x2)  # (B,C,d_model) 通过 FourierLayer

        # MLP 残差块
        x = x + self.mlp(self.ln2(x))  # (B,T,C)

        # 去均值操作
        m = torch.mean(x, dim=1, keepdim=True)  # (B,1,C)
        return x - m, self.linear(m), trend, season

    

class Decoder(nn.Module):
    """
    解码器：堆叠多个 DecoderBlock，并累积趋势/季节性与均值项。
    输入：
        - x: (B, T, C)
        - t: (B,) 或 (B,1)
        - enc: 编码器输出 (B, T_enc, C)
    输出：
        - x: (B, T, C)
        - mean: (B, n_layer * T, n_feat)
        - trend: (B, C, n_feat)
        - season: (B, C, d_model)
    """
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
        condition_dim=512,
        use_freq_attn=False,
        freq_size=None,
        freq_heads=4,
        freq_head_dim=16,
        freq_pdrop=0.1,
        freq_resid_pdrop=0.1,
    ):
        super().__init__()
        self.d_model = n_embd
        self.n_feat = n_feat
        # pass through freq-attn params from parent if present
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
                use_freq_attn=use_freq_attn,
                freq_size=freq_size,
                freq_heads=freq_heads,
                freq_head_dim=freq_head_dim,
                freq_pdrop=freq_pdrop,
                freq_resid_pdrop=freq_resid_pdrop,
        ) for _ in range(n_layer)])
      
    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        b, c, _ = x.shape  # (B, T, C)
        # att_weights = []
        mean = []
        season = torch.zeros((b, c, self.d_model), device=x.device)  # (B,C,d_model)
        trend = torch.zeros((b, c, self.n_feat), device=x.device)    # (B,C,n_feat)
        for block_idx in range(len(self.blocks)):
            x, residual_mean, residual_trend, residual_season = \
                self.blocks[block_idx](x, enc, t, mask=padding_masks, label_emb=label_emb)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)  # 在层维拼接 (B, n_layer, n_feat) 或 (B, n_layer*T, n_feat) 视映射而定
        return x, mean, trend, season


class Transformer(nn.Module):
    """
    顶层模型：
    - 编码器获得条件上下文，解码器在相同嵌入空间生成输出；
    - 同时组合趋势（trend）与季节性残差（season_error），以提升可解释性。

    输入/输出：
    - input: (B, C_in, L)
    - 返回: (trend: (B, C_in, L), season_error: (B, C_in, L))
    - 若 return_res=True，额外返回纯季节项与中心化残差。
    """
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

        # frequency-attention related args (from model params / kwargs)
        self.use_freq_attn = bool(kwargs.get('use_freq_attn', False))
        freq_heads = int(kwargs.get('freq_heads', kwargs.get('h_f', 4)))
        freq_head_dim = int(kwargs.get('freq_d_model', kwargs.get('d_f', 16)))
        freq_pdrop = float(kwargs.get('freq_dropout', 0.1))

    # Encoder / Decoder: pass frequency-attn params through to blocks
        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate,
                               use_freq_attn=self.use_freq_attn, freq_size=n_feat,
                               freq_heads=freq_heads, freq_head_dim=freq_head_dim,
                               freq_pdrop=freq_pdrop, freq_resid_pdrop=freq_pdrop)
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        self.decoder = Decoder(n_channel, n_feat, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times,
                               block_activate, condition_dim=n_embd,
                               use_freq_attn=self.use_freq_attn, freq_size=n_feat,
                               freq_heads=freq_heads, freq_head_dim=freq_head_dim,
                               freq_pdrop=freq_pdrop, freq_resid_pdrop=freq_pdrop)
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        # 便捷调试输出：在日志中明确频域注意力是否启用及其关键参数
        if self.use_freq_attn:
            print(f"[Transformer] Frequency-attention ENABLED: freq_heads={freq_heads}, head_dim={freq_head_dim}, F={n_feat}")
        else:
            print("[Transformer] Frequency-attention DISABLED")

    def forward(self, input, t, padding_masks=None, return_res=False):
        emb = self.emb(input)            # (B, C_in, L) -> (B, T, C) 经 Conv_MLP 到嵌入空间
        inp_enc = self.pos_enc(emb)      # 位置编码 (B, T, C)
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks)  # (B, T, C)

        inp_dec = self.pos_dec(emb)      # (B, T, C)
        output, mean, trend, season = self.decoder(inp_dec, t, enc_cond, padding_masks=padding_masks)

        res = self.inverse(output)       # (B, C_in, L)
        res_m = torch.mean(res, dim=1, keepdim=True)  # (B,1,L)
        # 季节性残差 = season 线性组合 + 残差去中心化
        season_error = self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        # 趋势 = 多层均值组合 + 残差均值 + 趋势分量
        trend = self.combine_m(mean) + res_m + trend

        if return_res:
            return trend, self.combine_s(season.transpose(1, 2)).transpose(1, 2), res - res_m

        return trend, season_error


if __name__ == '__main__':
    pass