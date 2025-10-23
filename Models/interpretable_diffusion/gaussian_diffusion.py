import math
import logging
import torch
import torch.nn.functional as F

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.interpretable_diffusion.transformer import Transformer
from Models.interpretable_diffusion.model_utils import default, identity, extract

# 设置模块级日志器
logger = logging.getLogger(__name__)


# gaussian diffusion trainer class

def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """
    线性 beta 调度。
    返回 dtype=float64 的张量以匹配全链路双精度。
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    余弦 beta 调度（https://openreview.net/forum?id=-NEXDKk8gZ）。
    返回 dtype=float64 的张量以匹配全链路双精度。
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion_TS(nn.Module):
    """
    Diffusion-TS：用于时间序列生成/插值/条件采样的扩散模型（Transformer 主干）。

    形状约定（全链路统一为 (B, T, C)）：
    - B: batch size
    - T: seq_length （时间步）
    - C: feature_size （特征/通道数）
    - 全链路 dtype 统一为 float64
    """

    def __init__(
        self,
        seq_length: int,
        feature_size: int,
        n_layer_enc: int = 3,
        n_layer_dec: int = 6,
        d_model: int | None = None,
        timesteps: int = 1000,
        sampling_timesteps: int | None = None,
        loss_type: str = 'l1',
        beta_schedule: str = 'cosine',
        n_heads: int = 4,
        mlp_hidden_times: int = 4,
        eta: float = 0.0,
        attn_pd: float = 0.0,
        resid_pd: float = 0.0,
        kernel_size: int | None = None,
        padding_size: int | None = None,
        use_fft: bool = False,            # 是否启用频域损失（CSI 无相位可关闭）
        reg_weight: float | None = None,  # 频域损失权重（use_fft=True 时生效）
        corr_weight: float = 0.1,         # λ：相关性正则的总权重
        # ---- 以下为抽出的“魔法常量”（可调） ----
        corr_time_weight: float = 0.5,    # μ：时间自相关在相关性正则中的权重
        corr_max_lag: int = 10,           # K：时间自相关的最大 lag
        corr_eps: float = 1e-8,           # 相关性计算用到的数值稳定项 ε
        loss_weight_divisor: float = 100, # 训练重加权分母常量（源自论文中的 reweight 缩放）
        sigma_eps: float = 1e-12,         # Langevin 步中判断 sigma 是否趋近 0 的阈值
        # Langevin 分阶段阈值（按 t/num_timesteps 的比例）
        langevin_phase_low: float = 0.05,
        langevin_phase_mid: float = 0.75,
        langevin_phase_high: float = 0.90,
        # Langevin 每个阶段的迭代步数 K 与学习率缩放
        langevin_K_low: int = 0,          # t 比例 < low
        langevin_K_mid: int = 1,          # low <= 比例 <= mid
        langevin_K_high: int = 2,         # mid < 比例 <= high
        langevin_K_very_high: int = 3,    # 比例 > high
        langevin_lr_scale_mid: float = 0.25,
        langevin_lr_scale_high: float = 0.5,
        **kwargs,
    ):
        """
        Diffusion-TS 模型初始化。
        - 统一 dtype 为 float64；
        - 支持关闭/启用频域一致性；
        - 引入相关性正则（通道相关 + 时间自相关）。
        """

        super(Diffusion_TS, self).__init__()

        # 兼容旧参数命名：use_ff -> use_fft
        if 'use_ff' in kwargs:
            use_fft = bool(kwargs.pop('use_ff'))

        # ========== 基本配置 ==========
        self.dtype = torch.float64
        self.eta = eta
        self.use_fft = use_fft
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.corr_weight = corr_weight  # λ，总体相关性权重

        # 相关性正则与数值稳定参数
        self._corr_eps = corr_eps
        self.corr_time_weight = corr_time_weight  # μ
        self.corr_max_lag = corr_max_lag         # K
        self.sigma_eps = sigma_eps

        # Langevin 阶段与超参（按 t 比例分段）
        self.langevin_phase_low = langevin_phase_low
        self.langevin_phase_mid = langevin_phase_mid
        self.langevin_phase_high = langevin_phase_high
        self.langevin_K_low = langevin_K_low
        self.langevin_K_mid = langevin_K_mid
        self.langevin_K_high = langevin_K_high
        self.langevin_K_very_high = langevin_K_very_high
        self.langevin_lr_scale_mid = langevin_lr_scale_mid
        self.langevin_lr_scale_high = langevin_lr_scale_high

        # Fourier loss 权重（若 use_fft=False，将不会被使用）
        self.fft_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)
        # 兼容旧属性名
        self.ff_weight = self.fft_weight

        # ========== Transformer 编码器/解码器 ==========
        self.model = Transformer(
            n_feat=feature_size,
            n_channel=seq_length,
            n_layer_enc=n_layer_enc,
            n_layer_dec=n_layer_dec,
            n_heads=n_heads,
            attn_pdrop=attn_pd,
            resid_pdrop=resid_pd,
            mlp_hidden_times=mlp_hidden_times,
            max_len=seq_length,
            n_embd=d_model,
            conv_params=[kernel_size, padding_size],
            **kwargs
        ).to(dtype=self.dtype)

        # ========== β 调度策略 ==========
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'Unknown beta schedule: {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.).to(self.dtype)

        timesteps_, = betas.shape
        self.num_timesteps = int(timesteps_)
        self.loss_type = loss_type

        # ========== 采样配置 ==========
        self.sampling_timesteps = int(default(sampling_timesteps, timesteps))
        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # ========== 注册缓冲参数（保持 float64） ==========
        register_buffer = lambda name, val: self.register_buffer(name, val.to(self.dtype))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 扩散相关计算参数
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # 后验分布计算参数
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # 损失重加权项（对应论文中 training reweighting），除数可配置
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / loss_weight_divisor)

        # ========== 调试信息 ==========
        logger.info(
            f"[Diffusion_TS] Initialized: seq_length={seq_length}, feature_size={feature_size}, "
            f"use_fft={use_fft}, corr_weight={corr_weight}, dtype=float64"
        )


    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def denoise_pred(self, x: torch.Tensor, t: torch.Tensor, padding_masks: torch.Tensor | None = None) -> torch.Tensor:
        """
        网络前向预测：给定 x_t 与 t，输出 x0 的估计（趋势+季节项）。

        输入/输出形状：
        - x: (B, T, C)
        - t: (B,)
        - padding_masks: (B, T) 布尔掩码，True 表示有效时间步
        - 返回: (B, T, C)
        """
        x = x.to(self.dtype)
        t = t.to(torch.long)
        trend, season = self.model(x, t, padding_masks=padding_masks)
        model_output = (trend + season).to(self.dtype)
        return model_output

    def model_predictions(self, x, t, clip_x_start=False, padding_masks=None):
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x = x.to(self.dtype)
        x_start = self.denoise_pred(x, t, padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True):
        _, x_start = self.model_predictions(x, t)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    def p_sample(self, x, t: int, clip_denoised=True, cond_fn=None, model_kwargs=None):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x, dtype=self.dtype) if t > 0 else 0.  # no noise if t == 0
        if cond_fn is not None:
            model_mean = self.condition_mean(
                cond_fn, model_mean, model_log_variance, x, t=batched_times, model_kwargs=model_kwargs
            )
        pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_series, x_start

    @torch.no_grad()
    def sample(self, shape):
        """
        逐步反向扩散采样。
        - shape: (B, T, C)
        - 返回: (B, T, C)
        """
        device = self.betas.device
        series = torch.randn(shape, device=device, dtype=self.dtype)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            series, _ = self.p_sample(series, t)
        return series

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        series = torch.randn(shape, device=device, dtype=self.dtype)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(series, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                series = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(series, dtype=self.dtype)
            series = x_start * alpha_next.sqrt() + \
                     c * pred_noise + \
                     sigma * noise

        return series
    
    def generate_mts(self, batch_size=16, model_kwargs=None, cond_fn=None):
        feature_size, seq_length = self.feature_size, self.seq_length
        if cond_fn is not None:
            sample_fn = self.fast_sample_cond if self.fast_sampling else self.sample_cond
            return sample_fn((batch_size, seq_length, feature_size), model_kwargs=model_kwargs, cond_fn=cond_fn)
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn((batch_size, seq_length, feature_size))

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start, dtype=self.dtype))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    # -------------------- Loss components (拆分提高可维护性) --------------------
    def _pointwise_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """点对点损失（L1/L2），返回与输入同形状的逐元素损失。
        - pred/target: (B, T, C)
        - 返回: (B, T, C)
        """
        return self.loss_fn(pred, target, reduction='none')

    def _fourier_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """频域损失：对实部和虚部分别计算 L1/L2 并相加。
        - 输入/输出: (B, T, C)
        - 当 use_fft=False 时返回全零张量。
        """
        if not self.use_fft:
            return torch.zeros_like(pred, dtype=self.dtype)
        # 在时间维 T 上做 FFT（将时间维移到最后以便 torch.fft.fft 沿最后一维）
        fft_pred = torch.fft.fft(pred.transpose(1, 2), norm='forward')  # (B, C, T) complex128
        fft_tgt = torch.fft.fft(target.transpose(1, 2), norm='forward')
        fft_pred = fft_pred.transpose(1, 2)  # (B, T, C)
        fft_tgt = fft_tgt.transpose(1, 2)
        loss_real = self.loss_fn(torch.real(fft_pred), torch.real(fft_tgt), reduction='none')
        loss_imag = self.loss_fn(torch.imag(fft_pred), torch.imag(fft_tgt), reduction='none')
        return self.fft_weight * (loss_real + loss_imag)

    def _correlation_regularizer(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """相关性正则：通道相关矩阵差 + 时间自相关差。
        - pred/target: (B, T, C)
        - 返回: (B,) 每个样本一个标量
        """
        B, T, C = pred.shape
        eps = self._corr_eps

        # 中心化（沿时间维 dim=1）
        Xc = pred - pred.mean(dim=1, keepdim=True)   # (B, T, C)
        Yc = target - target.mean(dim=1, keepdim=True)

        # ---- (1) 通道相关矩阵差：对每个样本计算 CxC 的相关矩阵并比较 ----
        Xc_ch = Xc.transpose(1, 2)  # (B, C, T)
        Yc_ch = Yc.transpose(1, 2)
        cov_X = torch.matmul(Xc_ch, Xc_ch.transpose(1, 2)) / float(max(T - 1, 1))  # (B, C, C)
        cov_Y = torch.matmul(Yc_ch, Yc_ch.transpose(1, 2)) / float(max(T - 1, 1))

        var_X = cov_X.diagonal(dim1=1, dim2=2)  # (B, C)
        var_Y = cov_Y.diagonal(dim1=1, dim2=2)  # (B, C)
        std_X = torch.sqrt(var_X.clamp(min=0.) + eps)  # (B, C)
        std_Y = torch.sqrt(var_Y.clamp(min=0.) + eps)
        denom_X = std_X.unsqueeze(2) * std_X.unsqueeze(1)  # (B, C, C)
        denom_Y = std_Y.unsqueeze(2) * std_Y.unsqueeze(1)
        corr_X = cov_X / (denom_X + eps)
        corr_Y = cov_Y / (denom_Y + eps)
        channel_corr_loss = F.l1_loss(corr_X, corr_Y, reduction='none')  # (B, C, C)
        channel_corr_loss = channel_corr_loss.view(B, -1).mean(dim=1)    # (B,)

        # ---- (2) 时间自相关差：lag ∈ [1, K] ----
        K = min(self.corr_max_lag, T - 1)
        if K <= 0:
            temporal_autocorr_loss = torch.zeros((B,), device=pred.device, dtype=self.dtype)
        else:
            # 使用通道方差估计（与上面的 std_X 一致），std_X: (B, C)
            temporal_losses = torch.zeros((B, C), device=pred.device, dtype=self.dtype)
            for k in range(1, K + 1):
                # numerator: sum_t X[t]*X[t+k] 按时间维聚合
                num_X = (Xc[:, :T - k, :] * Xc[:, k:, :]).sum(dim=1)  # (B, C)
                num_Y = (Yc[:, :T - k, :] * Yc[:, k:, :]).sum(dim=1)  # (B, C)
                denom = (T - k) * (std_X * std_X + eps)              # (B, C)
                ac_X = num_X / (denom + eps)
                ac_Y = num_Y / (denom + eps)
                temporal_losses += torch.abs(ac_X - ac_Y)
            temporal_autocorr_loss = temporal_losses.mean(dim=1) / float(K)  # (B,)

        # 组合
        corr_loss_batch = channel_corr_loss + self.corr_time_weight * temporal_autocorr_loss  # (B,)
        return corr_loss_batch

    def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None):
        """
        训练损失：点损失 + 可选 FFT 损失 + 相关性正则。

        输入/输出形状：
        - x_start: (B, T, C)
        - t: (B,)
        - target: (B, T, C) 若为 None，则等于 x_start
        - 返回：标量 loss
        """
        x_start = x_start.to(self.dtype)
        noise = default(noise, lambda: torch.randn_like(x_start, dtype=self.dtype))
        if target is None:
            target = x_start
        else:
            target = target.to(self.dtype)

        # 噪声采样与模型预测
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        pred = self.denoise_pred(x_t, t, padding_masks)

        # 基础点对点损失（逐元素）
        base_loss = self._pointwise_loss(pred, target)

        # 频域损失（逐元素）
        fft_loss = self._fourier_loss(pred, target)

        # 合并逐元素并按样本平均 -> (B,)
        per_elem = base_loss + fft_loss
        per_sample = reduce(per_elem, 'b ... -> b (...)', 'mean')  # (B, 1)

        # 相关性正则（每样本一个标量）
        corr_loss_batch = self._correlation_regularizer(pred, target)  # (B,)
        per_sample = per_sample + self.corr_weight * corr_loss_batch.unsqueeze(1)

        # 时间步重加权与 batch 平均
        per_sample = per_sample * extract(self.loss_weight, t, per_sample.shape)
        return per_sample.mean()

    def forward(self, x, **kwargs):
        """
        前向训练入口。
        - x: (B, T, C)
        - 返回: 标量 loss
        """
        B, T, C = x.shape
        assert C == self.feature_size, f'number of variable must be {self.feature_size}'
        device = x.device
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        return self._train_loss(x_start=x.to(self.dtype), t=t, **kwargs)

    def return_components(self, x, t: int):
        """
        返回分解组件（趋势、季节、残差）与对应的 x_t。
        - x: (B, T, C)
        - t: int
        - 返回: (trend, season, residual, x_t)
        """
        B, T, C = x.shape
        assert C == self.feature_size, f'number of variable must be {self.feature_size}'
        device = x.device
        t_vec = torch.tensor([t], device=device).repeat(B).to(torch.long)
        x_t = self.q_sample(x.to(self.dtype), t_vec)
        trend, season, residual = self.model(x_t, t_vec, return_res=True)
        return trend, season, residual, x_t

    def fast_sample_infill(self, shape, target, sampling_timesteps, partial_mask=None, clip_denoised=True, model_kwargs=None):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        series = torch.randn(shape, device=device, dtype=self.dtype)

        for time, time_next in tqdm(time_pairs, desc='conditional sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(series, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                series = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(series, dtype=self.dtype)

            series = pred_mean + sigma * noise
            series = self.langevin_fn(sample=series, mean=pred_mean, sigma=sigma, t=time_cond,
                                      tgt_embs=target.to(self.dtype), partial_mask=partial_mask, **model_kwargs)
            target_t = self.q_sample(target, t=time_cond)
            series[partial_mask] = target_t[partial_mask]

        series[partial_mask] = target[partial_mask]

        return series

    def sample_infill(
        self,
        shape, 
        target,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        series = torch.randn(shape, device=device, dtype=self.dtype)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='conditional sampling loop time step', total=self.num_timesteps):
            series = self.p_sample_infill(x=series, t=t, clip_denoised=clip_denoised, target=target,
                                          partial_mask=partial_mask, model_kwargs=model_kwargs)
        
        series[partial_mask] = target[partial_mask]
        return series
    
    def p_sample_infill(
        self,
        x,
        target,
        t: int,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None
    ):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x, dtype=self.dtype) if t > 0 else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self.langevin_fn(sample=pred_img, mean=model_mean, sigma=sigma, t=batched_times,
                                    tgt_embs=target.to(self.dtype), partial_mask=partial_mask, **model_kwargs)
        
        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]

        return pred_img

    def langevin_fn(
        self,
        coef,
        partial_mask,
        tgt_embs,
        learning_rate,
        sample,
        mean,
        sigma,
        t,
        coef_=0.
    ):
    
        # 按 t 比例（t/num_timesteps）分段控制 K 与学习率缩放
        t_ratio = t[0].item() / max(self.num_timesteps, 1)
        if t_ratio < self.langevin_phase_low:
            K = self.langevin_K_low
        elif t_ratio > self.langevin_phase_high:
            K = self.langevin_K_very_high
        elif t_ratio > self.langevin_phase_mid:
            K = self.langevin_K_high
            learning_rate = learning_rate * self.langevin_lr_scale_high
        else:
            K = self.langevin_K_mid
            learning_rate = learning_rate * self.langevin_lr_scale_mid

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.denoise_pred(x=input_embs_param, t=t)

                if sigma.mean() <= self.sigma_eps:
                    logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()
            
                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param = torch.nn.Parameter((input_embs_param.data + coef_ * sigma.mean().item() * epsilon).detach())

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample
    
    def condition_mean(self, cond_fn, mean, log_variance, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x=x, t=t, **model_kwargs)
        # 统一为 float64，避免不必要的 float32 转换
        gradient = gradient.to(self.dtype)
        mean = mean.to(self.dtype)
        log_variance = log_variance.to(self.dtype)
        new_mean = mean + torch.exp(log_variance) * gradient
        return new_mean
    
    def condition_score(self, cond_fn, x_start, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        eps = self.predict_noise_from_start(x, t, x_start)
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        pred_xstart = self.predict_start_from_noise(x, t, eps)
        model_mean, _, _ = self.q_posterior(x_start=pred_xstart, x_t=x, t=t)
        return model_mean, pred_xstart
    
    def sample_cond(
        self,
        shape,
        clip_denoised=True,
        model_kwargs=None,
        cond_fn=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        series = torch.randn(shape, device=device, dtype=self.dtype)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            series, x_start = self.p_sample(series, t, clip_denoised=clip_denoised, cond_fn=cond_fn,
                                            model_kwargs=model_kwargs)
        return series

    def fast_sample_cond(
        self,
        shape,
        clip_denoised=True,
        model_kwargs=None,
        cond_fn=None
    ):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        series = torch.randn(shape, device=device, dtype=self.dtype)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(series, time_cond, clip_x_start=clip_denoised)

            if cond_fn is not None:
                _, x_start = self.condition_score(cond_fn, x_start, series, time_cond, model_kwargs=model_kwargs)
                pred_noise = self.predict_noise_from_start(series, time_cond, x_start)

            if time_next < 0:
                series = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(series, dtype=self.dtype)
            series = x_start * alpha_next.sqrt() + \
                     c * pred_noise + \
                     sigma * noise

        return series


if __name__ == '__main__':
    pass
