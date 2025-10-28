import math
import torch
import torch.nn.functional as F

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.interpretable_diffusion.transformer import Transformer
from Models.interpretable_diffusion.model_utils import default, identity, extract


# gaussian diffusion trainer class

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion_TS(nn.Module):
    """
    时间序列扩散模型（Diffusion-TS）。
    主要作用：
    - 使用 Transformer 主干建模趋势与季节性并合成输出；
    - 构建扩散过程的调度参数，支持标准与快速采样；
    - 训练阶段可引入频域损失（可选）与相关性正则（通道相关 + 时间自相关）。

    Shape 约定（默认）：
    - Unless otherwise specified, tensors follow shape (B, T, C):
      B = batch size, T = seq_length (time), C = feature_size (channels/subcarriers).
    - 在某些统计计算（如通道相关矩阵）中，会显式说明临时使用 (B, C, T)。
    """
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
        """
        模型初始化：构建 Transformer 主体与扩散所需的缓冲张量与配置项。
        作用：
        - 初始化 Transformer（编码/解码）与必要的投影超参；
        - 根据 beta 调度（linear/cosine）生成扩散系数与其累计量；
        - 配置训练损失（点对点、可选的频域）与相关性正则。

        Inputs:
        - seq_length: int, 时间长度 T
        - feature_size: int, 特征/通道数 C
        - timesteps: int, 扩散总步数
        - sampling_timesteps: Optional[int], 采样步数（若 < timesteps 则为快速采样）
        - loss_type: str, 'l1' 或 'l2'
        - beta_schedule: str, 'linear' 或 'cosine'
        - use_ff: bool, 是否启用频域（FFT）损失
        - corr_weight: float, 相关性正则的权重

        Shapes:
        - 默认输入/输出: (B, T, C)
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
        # Debug print to verify frequency-attention flags are propagated
        if 'use_freq_attn' in kwargs:
            print(f"[Diffusion_TS] kwargs.use_freq_attn={kwargs.get('use_freq_attn')}, "
                  f"freq_heads={kwargs.get('freq_heads', kwargs.get('h_f'))}, "
                  f"freq_d_model={kwargs.get('freq_d_model', kwargs.get('d_f'))}, "
                  f"freq_dropout={kwargs.get('freq_dropout')}")
        else:
            print("[Diffusion_TS] No 'use_freq_attn' found in kwargs (model.params).")

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


    def predict_noise_from_start(self, x_t, t, x0):
        """
        由去噪起点 x0 与当前噪声步 x_t 反推噪声项。

        Inputs:
        - x_t: 当前步样本, (B, T, C)
        - t: 时间步, (B,)
        - x0: 预测的起点, (B, T, C)

        Returns:
        - noise: 估计噪声, (B, T, C)
        """
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        由 x_t 与噪声项重构去噪起点 x0。

        Inputs:
        - x_t: 当前步样本, (B, T, C)
        - t: 时间步, (B,)
        - noise: 噪声, (B, T, C)

        Returns:
        - x0: 起点估计, (B, T, C)
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        计算 q(x_{t-1} | x_t, x_0) 的后验均值与方差（扩散前向过程的逆推）。

        Inputs:
        - x_start: 起点 x0, (B, T, C)
        - x_t: 当前步样本, (B, T, C)
        - t: 时间步, (B,)

        Returns:
        - posterior_mean: (B, T, C)
        - posterior_variance: (B, T, C)
        - posterior_log_variance_clipped: (B, T, C)
        """
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def output(self, x, t, padding_masks=None):
        """
        前向主干：调用 Transformer 得到趋势与季节性并求和作为模型输出。

        Inputs:
        - x: 输入序列（可为加噪序列）, (B, T, C)
        - t: 时间步, (B,)
        - padding_masks: Optional[BoolTensor], (B, T)

        Returns:
        - model_output: (B, T, C)
        """
        trend, season = self.model(x, t, padding_masks=padding_masks)
        model_output = trend + season
        return model_output

    def model_predictions(self, x, t, clip_x_start=False, padding_masks=None):
        """
        预测去噪起点与对应噪声。

        Returns:
        - pred_noise: (B, T, C)
        - x_start: (B, T, C)
        """
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.output(x, t, padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True):
        """
        基于模型预测得到 p(x_{t-1} | x_t) 的均值和方差，并可选裁剪 x_0 估计。
        """
        _, x_start = self.model_predictions(x, t)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    def p_sample(self, x, t: int, clip_denoised=True, cond_fn=None, model_kwargs=None):
        """
        反向一步采样：x_t -> x_{t-1}

        Inputs:
        - x: 当前步样本, (B, T, C)
        - t: int, 当前时间步
        - cond_fn: Optional[Callable], 条件梯度函数 grad log p(y|x)

        Returns:
        - pred_series: 下一步样本, (B, T, C)
        - x_start: 起点估计, (B, T, C)
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
    def sample(self, shape):
        """
        标准逐步采样，从纯噪声开始生成样本。

        Inputs:
        - shape: Tuple, (B, T, C)
        """
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True):
        """
        快速采样（DDIM 类似），使用更稀疏的时间步序列进行生成。
        """
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
    
    def generate_mts(self, batch_size=16, model_kwargs=None, cond_fn=None):
        """
        生成多条时间序列样本，自动选择标准或快速采样，并可选条件函数。
        Returns: (B, T, C)
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
        """
        前向加噪：x_0 -> x_t。

        Inputs:
        - x_start: 起点 x0, (B, T, C)
        - t: 时间步, (B,)
        - noise: Optional, (B, T, C)
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _compute_pointwise_and_fourier_loss(self, model_out, target):
        """
        计算点对点损失（L1/L2），并在启用 use_ff 时叠加频域（FFT）损失。

        Inputs:
        - model_out: (B, T, C)
        - target: (B, T, C)

        Returns:
        - train_loss: 张量形式的逐元素/逐位置损失（不进行 batch 归约），shape 同 model_out
        """
        # 基础点对点损失（L1 或 L2）
        train_loss = self.loss_fn(model_out, target, reduction='none')

        # fourier loss 可选：对时间维进行 FFT（通过转置将时间维置于最后一维）
        if self.use_ff:
            # 在 FFT 段禁用 autocast 并转换为 FP32，避免 cuFFT 的 FP16 长度限制
            with torch.cuda.amp.autocast(enabled=False):
                mo32 = model_out.transpose(1, 2).to(torch.float32)
                ta32 = target.transpose(1, 2).to(torch.float32)
                fft1 = torch.fft.fft(mo32, norm='forward')
                fft2 = torch.fft.fft(ta32, norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            # fourier_loss 以 FP32 计算，再与原损失做 dtype 对齐
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none') \
                           + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss = train_loss + self.ff_weight * fourier_loss.to(train_loss.dtype)

        return train_loss

    def _compute_correlation_regularization(self, model_out, target, train_loss_device):
        """
        计算相关性正则项：通道相关矩阵差 + 时间自相关差。

        说明/约定：
        - 默认输入/输出整体约定为 (B, T, C)；本段为按时间统计通道相关的运算，注释中将视作 (B, C, T)。
        - 为保持现有行为，不改变任何张量维度顺序，仅沿用原有实现的计算方式。

        Returns:
        - corr_loss_batch: (B,) 每个样本的相关性损失
        """
        # ---------- 相关性正则项（时域） ----------
        corr_loss_batch = torch.zeros((model_out.shape[0],), device=train_loss_device)

        # Shapes for this block (as-is in current implementation):
        # model_out, target: (B, C, T) where C = channels, T = time
        B, C, T = model_out.shape
        eps = self._corr_eps

        # (1) 通道间相关矩阵差
        Xm = model_out      # (B, C, T)
        Ym = target         # (B, C, T)
        Xc = Xm - Xm.mean(dim=2, keepdim=True)
        Yc = Ym - Ym.mean(dim=2, keepdim=True)

        # 样本级协方差矩阵与相关系数
        cov_X = torch.matmul(Xc, Xc.transpose(1, 2)) / float(max(T - 1, 1))   # (B, C, C)
        cov_Y = torch.matmul(Yc, Yc.transpose(1, 2)) / float(max(T - 1, 1))   # (B, C, C)
        var_X = cov_X.diagonal(dim1=1, dim2=2)  # (B, C)
        var_Y = cov_Y.diagonal(dim1=1, dim2=2)  # (B, C)
        std_X = torch.sqrt(var_X.clamp(min=0.) + eps)  # (B, C)
        std_Y = torch.sqrt(var_Y.clamp(min=0.) + eps)
        denom_X = std_X.unsqueeze(2) * std_X.unsqueeze(1)  # (B, C, C)
        denom_Y = std_Y.unsqueeze(2) * std_Y.unsqueeze(1)  # (B, C, C)
        corr_X = cov_X / (denom_X + eps)
        corr_Y = cov_Y / (denom_Y + eps)
        channel_corr_loss = F.l1_loss(corr_X, corr_Y, reduction='none')  # (B, C, C)
        channel_corr_loss = channel_corr_loss.view(B, -1).mean(dim=1)    # (B,)

        # (2) 时间自相关差（逐通道、逐 lag）
        K = min(self.corr_max_lag, T - 1)
        if K > 0:
            temporal_losses = torch.zeros((B, C), device=train_loss_device)
            for k in range(1, K + 1):
                num_X = (Xc[:, :, :T - k] * Xc[:, :, k:]).sum(dim=2)  # (B, C)
                num_Y = (Yc[:, :, :T - k] * Yc[:, :, k:]).sum(dim=2)  # (B, C)
                denom = ((T - k) * (std_X * std_X + eps))            # (B, C)
                ac_X = num_X / (denom + eps)
                ac_Y = num_Y / (denom + eps)
                temporal_losses += torch.abs(ac_X - ac_Y)
            temporal_autocorr_loss = temporal_losses.mean(dim=1) / float(K)  # (B,)
        else:
            temporal_autocorr_loss = torch.zeros((B,), device=train_loss_device)

        corr_loss_batch = channel_corr_loss + self.corr_time_weight * temporal_autocorr_loss
        return corr_loss_batch

    def _finalize_loss(self, train_loss, t, corr_loss_batch):
        """
        将逐位置损失按样本均值化，叠加相关性损失权重后，乘以时间步重加权项并做 batch 均值。

        Returns:
        - 标量损失（Tensor scalar）
        """
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')  # per-sample mean
        train_loss = train_loss + self.corr_weight * corr_loss_batch.unsqueeze(1)  # shape (B, 1)
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None):
        """
        训练一步：在给定噪声步 t 下计算损失（点对点 + 可选频域 + 相关性正则）。

        默认输入/输出形状为 (B, T, C)。在通道相关/自相关的统计计算中，下文明确标注临时形状 (B, C, T)。
        """
        # 基本噪声采样与模型输出（保持原逻辑）
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start

        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # noise sample
        model_out = self.output(x, t, padding_masks)

        # 点对点 + 可选频域损失（不改逻辑）
        train_loss = self._compute_pointwise_and_fourier_loss(model_out, target)

        # 相关性正则（不改逻辑）
        corr_loss_batch = self._compute_correlation_regularization(
            model_out=model_out,
            target=target,
            train_loss_device=train_loss.device,
        )

        # 重加权与归约（不改逻辑）
        return self._finalize_loss(train_loss, t, corr_loss_batch)

    def forward(self, x, **kwargs):
        """
        训练入口：随机采样时间步 t 并计算一个训练批次的损失。
        """
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, **kwargs)

    def return_components(self, x, t: int):
        """
        返回在给定 t 下的分解组件（趋势、季节、残差）与加噪后样本。
        Returns: trend, season, residual, x_t
        """
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    def fast_sample_infill(self, shape, target, sampling_timesteps, partial_mask=None, clip_denoised=True, model_kwargs=None):
        """
        快速条件插值采样：仅在未观测位置进行生成，同时保持已知位置与目标一致。
        - partial_mask: Bool mask, True 表示使用目标值固定该位置。
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
        """
        条件插值（慢速）：逐步从噪声生成，并在每一步保持已知位置与目标一致。
        Generate samples from the model and yield intermediate samples from each timestep of diffusion.
        """
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
        """
        条件插值的单步更新（x_t -> x_{t-1}），对已知位置进行强制替换。
        """
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
        """
        Langevin 风格的条件更新：在未观测位置对样本进行梯度步与噪声扰动。

        说明：
        - K 的取值与学习率缩放依赖于时间步 t（含阈值与缩放“魔法数”），用于早期/中期/后期不同强度的条件收敛；
        - 不改变任何逻辑，仅在注释中明确此启发式策略的作用范围。
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
        """
        条件化均值：基于条件梯度 cond_fn 对均值进行修正（Sohl-Dickstein et al., 2015）。

        English note:
        Compute the mean for the previous step given grad log p(y|x) and log variance.
        """
        gradient = cond_fn(x=x, t=t, **model_kwargs)
        new_mean = (
            mean.float() + torch.exp(log_variance) * gradient.float()
        )
        return new_mean
    
    def condition_score(self, cond_fn, x_start, x, t, model_kwargs=None):
        """
        条件化分数：基于 cond_fn 对分数进行修正（Song et al., 2020）。
        English note: Condition the score function and recompute model mean.
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
        条件采样（标准）：使用 cond_fn 在每一步进行条件修正。
        Generate samples and yield intermediate results across all timesteps.
        """
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
        """
        条件采样（快速）：在稀疏时间步上进行条件修正并生成样本。
        """
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
