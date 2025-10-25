import math
import torch
import torch.nn.functional as F

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from typing import Optional, Tuple
from Models.interpretable_diffusion.transformer import Transformer
from Models.interpretable_diffusion.model_utils import default, identity, extract


# gaussian diffusion trainer class

def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """线性 beta 调度。

    参数:
        timesteps: 扩散总步数。

    返回:
        形状为 (timesteps,) 的张量。
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """余弦（cosine）beta 调度。

    参考: https://openreview.net/forum?id=-NEXDKk8gZ

    参数:
        timesteps: 扩散总步数。
        s: 偏移量，避免 cos 为 0。

    返回:
        形状为 (timesteps,) 的张量。
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=4,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            use_ff=False,          # 不使用 FFT（CSI 无相位）
            reg_weight=None,       # Fourier loss 权重（保留参数但不使用）
            corr_weight=0.1,       # 🔹新增：时间相关性正则项的权重
            **kwargs
    ):
        """Diffusion-TS 模型初始化。

        说明:
            支持标准扩散训练与（可选）加速采样；默认关闭 Fourier 约束（适配无相位的 CSI），
            并额外引入一个时间/通道相关性正则项以增强生成的结构性。

        参数:
            seq_length: 序列长度（time steps, 记为 T）。
            feature_size: 特征维度/通道数（variables/channels, 记为 C）。
            timesteps: 扩散过程的步数。
            sampling_timesteps: 采样时使用的步数（若小于 timesteps 则启用快速采样）。
            loss_type: 基础点对点损失类型，'l1' 或 'l2'。
            beta_schedule: beta 调度策略，'linear' 或 'cosine'。
            eta: DDIM 噪声系数。
            use_ff: 是否使用 Fourier loss（对 CSI 默认 False）。
            reg_weight: Fourier loss 权重（use_ff=False 时不生效）。
            corr_weight: 相关性正则项权重（越大越强调结构对齐）。
        """

        super(Diffusion_TS, self).__init__()

        # ========== 基本配置 ==========
        self.eta = eta
        self.use_ff = use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.corr_weight = corr_weight  # λ，总体相关性权重
        self._corr_eps = 1e-8
        self.corr_time_weight = 0.5     # μ，时间自相关在总相关性损失中的权重
        self.corr_max_lag = 10          # K，时间自相关最大 lag

        # Fourier loss 权重（若 use_ff=False，将不会被使用）
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)

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
        )

        # ========== β 调度策略 ==========
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'Unknown beta schedule: {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # ========== 采样配置 ==========
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # ========== 注册缓冲参数 ==========
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
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

        # 损失重加权项（对应论文中 training reweighting）
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

        # ========== 调试信息 ==========
        print(f"[Diffusion_TS] Initialized with seq_length={seq_length}, feature_size={feature_size}, "
            f"use_ff={use_ff}, corr_weight={corr_weight}")


    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """由 x0 反推噪声项 ε。

        参数:
            x_t: 噪声污染后的序列，shape (B, T, C)。
            t: 时间步索引，shape (B,)。
            x0: 估计的无噪序列，shape (B, T, C)。
        """
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """由噪声项 ε 反推 x0。

        参数:
            x_t: 噪声污染后的序列，shape (B, T, C)。
            t: 时间步索引，shape (B,)。
            noise: 噪声项 ε，shape (B, T, C)。
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算后验分布 q(x_{t-1} | x_t, x_0) 的参数。

        返回: (posterior_mean, posterior_variance, posterior_log_variance_clipped)
        """
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def output(self, x: torch.Tensor, t: torch.Tensor, padding_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """模型前向，返回趋势与季节项之和。

        参数:
            x: 输入序列，shape (B, T, C)
            t: 时间步，shape (B,)
            padding_masks: 可选的 padding 掩码，shape (B, T)
        """
        trend, season = self.model(x, t, padding_masks=padding_masks)
        model_output = trend + season
        return model_output

    def model_predictions(self, x: torch.Tensor, t: torch.Tensor, clip_x_start: bool = False, padding_masks: Optional[torch.Tensor] = None):
        """返回 (pred_noise, x_start)。

        说明:
            - 先通过模型得到 x_start，再推回噪声项。
            - 可选地将 x_start 限幅到 [-1, 1]。
        """
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.output(x, t, padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True):
        """返回 p(x_{t-1}|x_t) 的均值与方差等参数，并附带 x_start。

        返回: (model_mean, posterior_variance, posterior_log_variance, x_start)
        """
        _, x_start = self.model_predictions(x, t)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    def p_sample(self, x: torch.Tensor, t: int, clip_denoised: bool = True, cond_fn=None, model_kwargs=None):
        """单步从 p(x_{t-1}|x_t) 采样。

        支持传入 cond_fn 进行条件引导。
        """
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        if cond_fn is not None:
            model_mean = self.condition_mean(
                cond_fn, model_mean, model_log_variance, x, t=batched_times, model_kwargs=model_kwargs
            )
        pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_series, x_start

    @torch.no_grad()
    def sample(self, shape: Tuple[int, int, int]) -> torch.Tensor:
        """标准逐步采样，遍历所有扩散步。

        参数:
            shape: (B, T, C)
        """
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def fast_sample(self, shape: Tuple[int, int, int], clip_denoised: bool = True) -> torch.Tensor:
        """加速采样（DDIM 风格），使用更少的时间步进行采样。"""
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img
    
    def generate_mts(self, batch_size: int = 16, model_kwargs=None, cond_fn=None) -> torch.Tensor:
        """生成多变量时间序列样本。

        参数:
            batch_size: 生成样本的批大小。
            model_kwargs: 传入条件函数的额外参数。
            cond_fn: 条件引导函数，可为 None。
        """
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
        """前向扩散 q(x_t | x_0)。

        参数:
            x_start: 原始无噪序列，shape (B, T, C)
            t: 时间步，shape (B,)
            noise: 外部提供的噪声，若为 None 则内部采样。
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # ========= Loss helpers for readability & testability =========
    def _pointwise_loss(self, model_out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """基础点对点损失（L1/L2），返回逐元素损失。

        Returns: (B, T, C)
        """
        return self.loss_fn(model_out, target, reduction='none')

    def _fourier_loss(self, model_out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Fourier 域损失（可选）。

        - 输入为 (B, T, C)，对时间维做 FFT。
        - 若未启用 use_ff，则返回全 0 的张量（与 pointwise loss 形状一致）。

        Returns: (B, T, C)
        """
        if not self.use_ff:
            return torch.zeros_like(model_out)
        from torch.cuda.amp import autocast
        with autocast(enabled=False):
            fft1 = torch.fft.fft(model_out.transpose(1, 2).float(), norm='forward')  # (B, C, T)
            fft2 = torch.fft.fft(target.transpose(1, 2).float(), norm='forward')     # (B, C, T)
        fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)  # -> (B, T, C)
        freq_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none') \
                    + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
        return freq_loss

    def _correlation_regularizer(self, model_out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """相关性正则（通道相关 + 时间自相关）。

        Returns:
            corr_loss_batch: shape (B,) 每个样本的相关性损失
        """
        # Shapes convention for correlation block: (B, C, T)
        # 原始张量通常为 (B, T, C)，此处统一转为 (B, C, T)
        if model_out.shape[1] == self.seq_length and model_out.shape[2] == self.feature_size:
            Xm = model_out.permute(0, 2, 1).contiguous()  # (B, C, T)
            Ym = target.permute(0, 2, 1).contiguous()     # (B, C, T)
        else:
            Xm, Ym = model_out, target

        B, C, T = Xm.shape
        eps = self._corr_eps

        # 1) 通道间相关矩阵差
        Xc = Xm - Xm.mean(dim=2, keepdim=True)
        Yc = Ym - Ym.mean(dim=2, keepdim=True)
        cov_X = torch.matmul(Xc, Xc.transpose(1, 2)) / float(max(T - 1, 1))  # (B, C, C)
        cov_Y = torch.matmul(Yc, Yc.transpose(1, 2)) / float(max(T - 1, 1))
        var_X = cov_X.diagonal(dim1=1, dim2=2)  # (B, C)
        var_Y = cov_Y.diagonal(dim1=1, dim2=2)  # (B, C)
        std_X = torch.sqrt(var_X.clamp(min=0.) + eps)
        std_Y = torch.sqrt(var_Y.clamp(min=0.) + eps)
        denom_X = std_X.unsqueeze(2) * std_X.unsqueeze(1)  # (B, C, C)
        denom_Y = std_Y.unsqueeze(2) * std_Y.unsqueeze(1)
        corr_X = cov_X / (denom_X + eps)
        corr_Y = cov_Y / (denom_Y + eps)
        channel_corr_loss = F.l1_loss(corr_X, corr_Y, reduction='none').view(B, -1).mean(dim=1)  # (B,)

        # 2) 时间自相关差
        K = min(self.corr_max_lag, T - 1)
        if K > 0:
            temporal_losses = torch.zeros((B, C), device=model_out.device)
            for k in range(1, K + 1):
                num_X = (Xc[:, :, :T - k] * Xc[:, :, k:]).sum(dim=2)  # (B, C)
                num_Y = (Yc[:, :, :T - k] * Yc[:, :, k:]).sum(dim=2)  # (B, C)
                denom = ((T - k) * (std_X * std_X + eps))
                ac_X = num_X / (denom + eps)
                ac_Y = num_Y / (denom + eps)
                temporal_losses += torch.abs(ac_X - ac_Y)
            temporal_autocorr_loss = temporal_losses.mean(dim=1) / float(K)  # (B,)
        else:
            temporal_autocorr_loss = torch.zeros((B,), device=model_out.device)

        corr_loss_batch = channel_corr_loss + self.corr_time_weight * temporal_autocorr_loss
        return corr_loss_batch  # (B,)

    def _finalize_loss(self, per_element_loss: torch.Tensor, corr_loss_batch: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """聚合损失：先样本内均值，再加上相关性项，最后做时间步重加权并聚合为标量。

        per_element_loss: (B, T, C)
        corr_loss_batch: (B,)
        """
        train_loss = reduce(per_element_loss, 'b ... -> b (...)', 'mean')  # -> (B, 1)
        train_loss = train_loss + self.corr_weight * corr_loss_batch.unsqueeze(1)
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None):
        """训练阶段的损失函数（重构版）。

        组成:
            1) 基础点损失（L1/L2）
            2) 可选 Fourier 约束（默认关闭）
            3) 相关性正则（通道间相关 + 时间自相关）
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start

        # 前向扩散 + 模型输出
        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # (B, T, C)
        model_out = self.output(x, t, padding_masks)          # (B, T, C)

        # 点损失 +（可选）Fourier 损失
        point_loss = self._pointwise_loss(model_out, target)                 # (B, T, C)
        fourier_loss = self._fourier_loss(model_out, target)                 # (B, T, C) or zeros
        total_pointwise = point_loss + (self.ff_weight * fourier_loss)       # (B, T, C)

        # 相关性正则 (B,)
        corr_loss_batch = self._correlation_regularizer(model_out, target)

        # 聚合
        return self._finalize_loss(total_pointwise, corr_loss_batch, t)

    def forward(self, x: torch.Tensor, **kwargs):
        """前向训练入口：随机采样 t 并返回损失。"""
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, **kwargs)

    def return_components(self, x: torch.Tensor, t: int):
        """返回在给定 t 下的趋势项、季节项、残差及带噪输入。

        返回: (trend, season, residual, x_t)
        """
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    def fast_sample_infill(self, shape: Tuple[int, int, int], target: torch.Tensor, sampling_timesteps: int, partial_mask: Optional[torch.Tensor] = None, clip_denoised: bool = True, model_kwargs=None):
        """加速条件填充（infill）。

        参数:
            shape: 输出形状 (B, T, C)
            target: 目标已知片段，shape (B, T, C)
            sampling_timesteps: 采样步数
            partial_mask: 已知片段的布尔掩码，shape (B, T, C) 或可广播到该形状
        """
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='conditional sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise
            img = self.langevin_fn(sample=img, mean=pred_mean, sigma=sigma, t=time_cond,
                                   tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
            target_t = self.q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]

        return img

    def sample_infill(
        self,
        shape, 
        target,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """标准逐步条件填充（infill）。"""
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='conditional sampling loop time step', total=self.num_timesteps):
            img = self.p_sample_infill(x=img, t=t, clip_denoised=clip_denoised, target=target,
                                       partial_mask=partial_mask, model_kwargs=model_kwargs)
        
        img[partial_mask] = target[partial_mask]
        return img
    
    def p_sample_infill(
        self,
        x,
        target,
        t: int,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None
    ):
        """单步条件填充（infill）。"""
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self.langevin_fn(sample=pred_img, mean=model_mean, sigma=sigma, t=batched_times,
                                    tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
        
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
        """Langevin 动力学更新已知/未知片段。

        参数说明（需结合外部传入的 model_kwargs 使用）:
            coef: 对 logp 项的权重。
            learning_rate: 对输入进行优化时的学习率。
            partial_mask: 已知片段的掩码，True/1 表示已知。
            tgt_embs: 目标序列（带监督的片段）。
            sample/mean/sigma/t: 当前步的采样、中间均值、方差与时间步。
        """
    
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=input_embs_param, t=t)

                if sigma.mean() == 0:
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
        """条件引导（Sohl-Dickstein et al., 2015）方式修正均值项。"""
        gradient = cond_fn(x=x, t=t, **model_kwargs)
        new_mean = (
            mean.float() + torch.exp(log_variance) * gradient.float()
        )
        return new_mean
    
    def condition_score(self, cond_fn, x_start, x, t, model_kwargs=None):
        """条件引导（Song et al., 2020）方式修正 score。"""
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
        """标准逐步采样（带条件引导）。"""
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(img, t, clip_denoised=clip_denoised, cond_fn=cond_fn,
                                         model_kwargs=model_kwargs)
        return img

    def fast_sample_cond(
        self,
        shape,
        clip_denoised=True,
        model_kwargs=None,
        cond_fn=None
    ):
        """加速采样（带条件引导）。"""
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if cond_fn is not None:
                _, x_start = self.condition_score(cond_fn, x_start, img, time_cond, model_kwargs=model_kwargs)
                pred_noise = self.predict_noise_from_start(img, time_cond, x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img


if __name__ == '__main__':
    pass
