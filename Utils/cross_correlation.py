import torch
from torch import nn


def cacf_torch(x, max_lag, dim=(0, 1)):
    """
    保留原实现：按最后一维（特征/子载波）两两成对，
    计算时间维上的互相关（lag 从 0 ... max_lag-1）。
    输入 x 形状约定为 (B, T, C)。
    返回形状: (B, max_lag, num_pairs)
    """
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    ind = get_lower_triangular_indices(x.shape[2])
    # 统一标准化（与原实现一致）：跨 (batch,time) 归一以稳定统计
    x = (x - x.mean(dim, keepdims=True)) / (x.std(dim, keepdims=True) + 1e-8)
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]
    cacf_list = []
    for i in range(max_lag):
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
        cacf_i = torch.mean(y, dim=1)  # 平均时间维
        cacf_list.append(cacf_i)
    cacf = torch.stack(cacf_list, dim=1)  # (B, max_lag, num_pairs)
    return cacf


def time_acf_torch(x: torch.Tensor, max_lag: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """
    计算沿时间维的自相关（per feature per batch）。
    输入: x (B, T, C)
    返回: (B, C, max_lag) 对应 lag=1..max_lag 的自相关。
    """
    B, T, C = x.shape
    # 每个样本、每个特征在时间维做标准化
    x_c = x - x.mean(dim=1, keepdim=True)
    std = x_c.std(dim=1, keepdim=True) + eps
    x_n = x_c / std

    ac_list = []
    for k in range(1, max_lag + 1):
        # (B, T-k, C) * (B, T-k, C) -> mean over time
        v = (x_n[:, :-k, :] * x_n[:, k:, :]).mean(dim=1)
        ac_list.append(v)  # (B, C)
    ac = torch.stack(ac_list, dim=2)  # (B, C, K)
    return ac


def freq_corr_matrix_torch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算沿时间维聚合得到的“子载波/特征间”相关矩阵。
    输入: x (B, T, C)
    返回: corr (B, C, C)
    """
    B, T, C = x.shape
    # 标准化：按时间维对每个样本-特征做零均值单位方差
    x_c = x - x.mean(dim=1, keepdim=True)
    std = x_c.std(dim=1, keepdim=True) + eps
    x_n = x_c / std  # (B, T, C)

    # 相关矩阵 ~ (X^T X) / T  (因 x_n 已标准化，协方差即相关)
    # (B, C, T) @ (B, T, C) -> (B, C, C)
    corr = torch.matmul(x_n.transpose(1, 2), x_n) / max(T, 1)
    return corr


class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


class CrossCorrelLoss(Loss):
    def __init__(self, x_real, lambda_time: float = 0.5, lambda_freq: float = 0.5,
                 max_lag_time: int = 1, **kwargs):
        """
        扩展为双通道相关性损失：
        L = λ_t * L_time + λ_f * L_freq
        - L_time: 沿时间维的自相关（每个特征的 ACF at lags 1..K），与真实的差异（L1）
        - L_freq: 子载波/特征之间的相关矩阵差异（L1）
        默认 λ_t = λ_f = 0.5，max_lag_time=1
        """
        super(CrossCorrelLoss, self).__init__(norm_foo=lambda x: torch.abs(x).sum(0), **kwargs)
        self.lambda_time = float(lambda_time)
        self.lambda_freq = float(lambda_freq)
        self.max_lag_time = int(max_lag_time)

        x_r = self.transform(x_real)
        # 预计算真实的时间自相关（按 batch 均值）与频率相关矩阵（按 batch 均值）
        # time ACF: (B, C, K) -> (C, K)
        self.tacf_real = time_acf_torch(x_r, max_lag=self.max_lag_time).mean(dim=0)
        # freq Corr: (B, C, C) -> (C, C)
        self.corr_real = freq_corr_matrix_torch(x_r).mean(dim=0)

    def compute(self, x_fake):
        x_f = self.transform(x_fake)
        device = x_f.device
        # 假样本的统计
        tacf_fake = time_acf_torch(x_f, max_lag=self.max_lag_time).mean(dim=0)   # (C, K)
        corr_fake = freq_corr_matrix_torch(x_f).mean(dim=0)                      # (C, C)

        # L_time：ACF 差异（L1 平均）
        l_time = torch.mean(torch.abs(tacf_fake - self.tacf_real.to(device)))

        # L_freq：相关矩阵差异（下三角含对角，避免重复计数）
        C = corr_fake.shape[0]
        tril_idx = torch.tril_indices(C, C, device=device)
        diff_tril = (corr_fake - self.corr_real.to(device))[tril_idx[0], tril_idx[1]]
        l_freq = torch.mean(torch.abs(diff_tril))

        # 组合与缩放（为了与旧实现数量级接近，这里保留 /10 的缩放）
        loss = self.lambda_time * l_time + self.lambda_freq * l_freq
        return loss