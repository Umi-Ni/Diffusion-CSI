import os
import sys
import csv
import time
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output, display
from torch.cuda.amp import autocast, GradScaler
from Utils.io_utils import instantiate_from_config, get_model_parameters_info

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])
        self.dataloader = dataloader['dataloader']
        self.step = 0
        self.milestone = 0
        self.args, self.config = args, config
        self.logger = logger
        # AMP (fp16) enable only on CUDA
        self.amp_enabled = (self.model.betas.device.type == 'cuda')
        self.scaler = GradScaler(enabled=self.amp_enabled)

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))
    
    def save_classifier(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current classifer to {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        data = {
            'step': self.step_classifier,
            'classifier': self.classifier.state_dict()
        }
        torch.save(data, str(self.results_folder / f'ckpt_classfier-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def load_classifier(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'ckpt_classfier-{milestone}.pt'), map_location=device)
        self.classifier.load_state_dict(data['classifier'])
        self.step_classifier = data['step']
        self.milestone_classifier = milestone

    def train(self):
        device = self.device
        step = 0
        loss_log = []  # 保存loss记录
        csv_path = os.path.join(self.results_folder, "loss_log.csv")
        os.makedirs(self.results_folder, exist_ok=True)

        # 初始化CSV文件
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])

        if self.logger is not None:
            tic = time.time()
            self.logger.log_info(f"{self.args.config_path}: start training...", check_primary=False)

        plt.ion()  # 开启交互式绘图模式（Notebook 实时刷新）

        # 定义滚动平均函数
        def rolling_average(data, window_size=50):
            data = np.array(data)
            if len(data) < window_size:
                return data  # 如果数据太短，直接返回原始
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        # 创建固定figure用于实时更新
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_display = display(fig, display_id=True)
        plt.close(fig)  # 初始关闭，避免多余显示

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.0

                # === 梯度累积 ===
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    # 使用 AMP 进行前向与反向
                    with autocast(enabled=self.amp_enabled, dtype=torch.float16):
                        loss = self.model(data, target=data)
                        loss = loss / self.gradient_accumulate_every

                    # 先缩放再反传，防止梯度下溢
                    self.scaler.scale(loss).backward()
                    total_loss += loss.item()

                # === 优化器更新 ===
                # 反缩放后再做梯度裁剪
                self.scaler.unscale_(self.opt)
                clip_grad_norm_(self.model.parameters(), 1.0)

                # AMP 更新优化器与 scaler
                self.scaler.step(self.opt)
                self.scaler.update()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                # === 记录loss ===
                loss_log.append(total_loss)
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([step, total_loss])

                # === tqdm显示 ===
                pbar.set_description(f"loss: {total_loss:.6f}")
                pbar.update(1)

                # === 实时绘图（每10步刷新一次）===
                if step % 10 == 0:
                    ax.clear()
                    
                    # 原始损失曲线（浅色、细线、半透明）
                    raw_steps = np.arange(1, len(loss_log) + 1)
                    ax.plot(raw_steps, loss_log, color="tab:blue", linewidth=1, alpha=0.5, label="Raw Loss")
                    
                    # 平滑损失曲线（深色、粗线）
                    window_size = 50  # 可调整，例如根据总步数动态设置，如 min(50, len(loss_log)//10)
                    smooth_loss = rolling_average(loss_log, window_size)
                    if len(smooth_loss) > 0:
                        smooth_steps = np.arange(len(loss_log) - len(smooth_loss) + 1, len(loss_log) + 1)
                        ax.plot(smooth_steps, smooth_loss, color="tab:red", linewidth=2, label="Smoothed Loss (window=50)")
                    
                    ax.set_xlabel("Step")
                    ax.set_ylabel("Loss")
                    ax.set_title("Training Loss (Real-time)")
                    ax.set_yscale('log')  # 新增：设置 y 轴为对数尺度
                    ax.grid(True)
                    ax.legend(loc="upper right")  # 添加图例
                    fig.tight_layout()
                    
                    plot_display.update(fig)

                # === 定期保存模型 ===
                if self.step != 0 and self.step % self.save_cycle == 0:
                    self.milestone += 1
                    self.save(self.milestone)

                # === 记录日志 ===
                if self.logger is not None and self.step % self.log_frequency == 0:
                    self.logger.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=self.step)

        # === 训练完成 ===
        plt.ioff()
        final_fig = plt.figure(figsize=(8, 5))
        
        # 原始损失曲线
        raw_steps = np.arange(1, len(loss_log) + 1)
        plt.plot(raw_steps, loss_log, color="blue", linewidth=1, alpha=0.5, label="Raw Loss")
        
        # 平滑损失曲线
        window_size = 50  # 与实时一致
        smooth_loss = rolling_average(loss_log, window_size)
        if len(smooth_loss) > 0:
            smooth_steps = np.arange(len(loss_log) - len(smooth_loss) + 1, len(loss_log) + 1)
            plt.plot(smooth_steps, smooth_loss, color="red", linewidth=2, label="Smoothed Loss (window=50)")
        
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Final Training Loss Curve")
        plt.yscale('log')  # 新增：设置 y 轴为对数尺度
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        final_path = os.path.join(self.results_folder, "loss_curve.png")
        plt.savefig(final_path, dpi=300)
        plt.close(final_fig)

        print(f"✅ Training complete! Loss curve saved to {final_path}")
        print(f"✅ CSV log saved to {csv_path}")

        if self.logger is not None:
            self.logger.log_info(f"Training done, time: {time.time() - tic:.2f}s")


    def sample(self, num, size_every, shape=None, model_kwargs=None, cond_fn=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every, model_kwargs=model_kwargs, cond_fn=cond_fn)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])
        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
        # return samples

    def forward_sample(self, x_start):
       b, c, h = x_start.shape
       noise = torch.randn_like(x_start, device=self.device)
       t = torch.randint(0, self.model.num_timesteps, (b,), device=self.device).long()
       x_t = self.model.q_sample(x_start=x_start, t=t, noise=noise).detach()
       return x_t, t

    def train_classfier(self, classifier):
        device = self.device
        step = 0
        self.milestone_classifier = 0
        self.step_classifier = 0
        dataloader = self.dataloader
        dataloader.dataset.shift_period('test')
        dataloader = cycle(dataloader)

        self.classifier = classifier
        self.opt_classifier = Adam(filter(lambda p: p.requires_grad, self.classifier.parameters()), lr=5.0e-4)
        scaler_cls = GradScaler(enabled=self.amp_enabled)
        
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training classifier...'.format(self.args.name), check_primary=False)
        
        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    x, y = next(dataloader)
                    x, y = x.to(device), y.to(device)
                    x_t, t = self.forward_sample(x)
                    with autocast(enabled=self.amp_enabled, dtype=torch.float16):
                        logits = classifier(x_t, t)
                        loss = F.cross_entropy(logits, y)
                        loss = loss / self.gradient_accumulate_every
                    scaler_cls.scale(loss).backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                scaler_cls.unscale_(self.opt_classifier)
                clip_grad_norm_(self.classifier.parameters(), 1.0)
                scaler_cls.step(self.opt_classifier)
                scaler_cls.update()
                self.opt_classifier.zero_grad()
                self.step_classifier += 1
                step += 1

                with torch.no_grad():
                    if self.step_classifier != 0 and self.step_classifier % self.save_cycle == 0:
                        self.milestone_classifier += 1
                        self.save_classfier(self.milestone_classifier)
                                            
                    if self.logger is not None and self.step_classifier % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

        # return classifier

