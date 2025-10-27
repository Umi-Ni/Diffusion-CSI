import math
import torch
from torch import nn
from einops import rearrange, reduce, repeat

class FourierLayer(nn.Module):
    """
    季节性建模（频域）：通过 rFFT 提取主要频率分量并回投到时域，近似长期周期性成分。

    参数：
    - d_model: 特征维度（与输入最后一维对齐）
    - low_freq: 去除的低频起始索引（用于略过直流项等）
    - factor: 选择频率个数的对数比例因子，top_k = int(factor * log(F))

    Shapes:
    - x: (B, T, D)
    - return: (B, T, D)
    """
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """季节性前向：在时间维执行 rFFT 并保留主频分量，回到时域。
        x: (B, T, D) -> seasonal (B, T, D)
        """
        # 为规避 cuFFT 对 FP16 的限制（长度需为 2 的幂），在 FFT 段强制使用 FP32 并禁用 autocast
        orig_dtype = x.dtype
        b, t, d = x.shape
        with torch.cuda.amp.autocast(enabled=False):
            x32 = x.to(torch.float32)
            x_freq = torch.fft.rfft(x32, dim=1)  # (B, F_rfft, D), complex64

            if t % 2 == 0:
                x_freq = x_freq[:, self.low_freq:-1]
                f = torch.fft.rfftfreq(t)
                f = f[self.low_freq:-1]
            else:
                x_freq = x_freq[:, self.low_freq:]
                f = torch.fft.rfftfreq(t)
                f = f[self.low_freq:]

            x_freq, index_tuple = self.topk_freq(x_freq)  # 选取幅值最大的 top-k 频率
            # 准备频率索引（在 CPU 上生成，再搬到目标设备）
            f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
            f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)  # (B, F_k, 1, D)
            seasonal32 = self.extrapolate(x_freq, f, t)  # FP32 时域合成

        return seasonal32.to(orig_dtype)

    def extrapolate(self, x_freq, f, t):
        """将频域选出的分量回到时域：幅值-相位形式合成余弦波并求和。
        x_freq: (B, F_k, D), f: (B, F_k, 1, D), t: int -> (B, T, D)
        """
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)  # 拼接共轭以对称化
        f = torch.cat([f, -f], dim=1)  # 频率正负对称
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)  # (1,1,T,1)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')   # (B,F,1,D)
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)  # (B,F,T,D)
        return reduce(x_time, 'b f t d -> b t d', 'sum')  # (B,T,D)

    def topk_freq(self, x_freq):
        """选择幅值最大的 top-k 频率分量。
        x_freq: (B, F, D) -> (B, F_k, D), index_tuple 用于回填采样索引。
        """
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))  # (B,1,F_k) 索引组合
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple
