import torch
from torch import nn
from Models.interpretable_diffusion.model_utils import Transpose

class TrendBlock(nn.Module):
    """
    趋势建模模块：使用多项式基函数对时间序列的长期趋势进行回归建模。

    意图：
    - 通过 1D 卷积提取通道间信息，并在固定多项式基底上回归趋势系数。
    - 输出与输入时间长度一致的趋势项，后续用于与季节性/残差组合。

    Shapes:
    - input: (B, C_in, L)
    - output: (B, C_out, L)  # 保持时序长度不变
    """
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        b, c, h = input.shape  # input: (B, C_in, L)
        x = self.trend(input).transpose(1, 2)  # trend conv stack -> (B, L, out_feat)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))  # (B, out_feat, L)
        trend_vals = trend_vals.transpose(1, 2)  # -> (B, L, out_feat)
        return trend_vals
