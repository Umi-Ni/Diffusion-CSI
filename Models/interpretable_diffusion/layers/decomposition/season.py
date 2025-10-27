import numpy as np
import torch
from torch import nn

class SeasonBlock(nn.Module):
    """
    季节性建模（时域基底）：使用 Fourier 基函数（cos/sin）进行季节项线性组合。

    Shapes:
    - input: (B, C_in, L)
    - output: (B, C_out, L)
    """
    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 \
            else (season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        b, c, h = input.shape  # (B, C_in, L)
        x = self.season(input)  # (B, season_poly, L)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))  # (B, L, season_poly) x (season_poly, L)
        season_vals = season_vals.transpose(1, 2)  # -> (B, season_poly, L)
        return season_vals
