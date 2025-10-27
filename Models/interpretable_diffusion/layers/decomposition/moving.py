import torch
from torch import nn
from Models.interpretable_diffusion.model_utils import series_decomp

class MovingBlock(nn.Module):
    """
    移动平均分解模块：
    - 使用滑动窗口分解输入序列为 (季节/残差, 趋势) 两部分。
    - 用于提供另一种可解释的趋势近似（与多项式回归互补）。

    Shapes:
    - input: (B, C_in, L)
    - return: (x: 残差近似, trend_vals: 趋势), 形状均为 (B, C_in, L)
    """
    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape  # (B, C, L)
        x, trend_vals = self.decomp(input)  # both: (B, C, L)
        return x, trend_vals
