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
        loss_log = []
        csv_path = os.path.join(self.results_folder, "loss_log.csv")
        os.makedirs(self.results_folder, exist_ok=True)

        # prepare csv file once and reuse handle for appends
        csv_file = open(csv_path, mode="w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "loss"])

        if self.logger is not None:
            tic = time.time()
            self.logger.log_info(f"{getattr(self.args, 'config_path', '')}: start training...", check_primary=False)

        plt.ion()

        def rolling_average(data, window_size=50):
            data = np.asarray(data)
            if data.size == 0:
                return np.array([])
            if data.size < window_size:
                return data.copy()
            kernel = np.ones(window_size, dtype=float) / window_size
            return np.convolve(data, kernel, mode="valid")

        fig, ax = plt.subplots(figsize=(7, 4))

        try:
            with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
                while self.step < self.train_num_steps:
                    total_loss = 0.0

                    # gradient accumulation
                    for _ in range(self.gradient_accumulate_every):
                        data = next(self.dl).to(device)
                        loss = self.model(data, target=data)
                        loss = loss / self.gradient_accumulate_every
                        loss.backward()
                        total_loss += loss.item()

                    # optimizer step
                    clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.sch.step(total_loss)
                    self.opt.zero_grad()
                    self.step += 1
                    self.ema.update()

                    # record loss
                    loss_log.append(total_loss)
                    csv_writer.writerow([self.step, total_loss])
                    csv_file.flush()

                    pbar.set_description(f"loss: {total_loss:.6f}")
                    pbar.update(1)

                    # update plot every 10 steps
                    if self.step % 10 == 0:
                        ax.clear()
                        raw_steps = np.arange(1, len(loss_log) + 1)
                        ax.plot(raw_steps, loss_log, color="tab:blue", linewidth=1, alpha=0.5, label="Raw Loss")

                        window_size = min(50, max(1, len(loss_log)//10))
                        smooth_loss = rolling_average(loss_log, window_size)
                        if smooth_loss.size > 0:
                            smooth_steps = np.arange(len(loss_log) - len(smooth_loss) + 1, len(loss_log) + 1)
                            ax.plot(smooth_steps, smooth_loss, color="tab:red", linewidth=2, label=f"Smoothed (w={window_size})")

                        ax.set_xlabel("Step")
                        ax.set_ylabel("Loss")
                        ax.set_title("Training Loss (Real-time)")
                        ax.set_yscale('log')
                        ax.grid(True)
                        ax.legend(loc="upper right")
                        fig.tight_layout()
                        # portable update
                        fig.canvas.draw()
                        plt.pause(0.001)

                    # periodic save
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=self.step)
        finally:
            csv_file.close()

        # === 训练完成 ===
        plt.ioff()
        final_fig = plt.figure(figsize=(8, 5))
        raw_steps = np.arange(1, len(loss_log) + 1)
        plt.plot(raw_steps, loss_log, color="blue", linewidth=1, alpha=0.5, label="Raw Loss")

        window_size = min(50, max(1, len(loss_log)//10))
        smooth_loss = rolling_average(loss_log, window_size)
        if smooth_loss.size > 0:
            smooth_steps = np.arange(len(loss_log) - len(smooth_loss) + 1, len(loss_log) + 1)
            plt.plot(smooth_steps, smooth_loss, color="red", linewidth=2, label=f"Smoothed (w={window_size})")

        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Final Training Loss Curve")
        plt.yscale('log')
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        final_path = os.path.join(self.results_folder, "loss_curve.png")
        plt.savefig(final_path, dpi=300)
        plt.close(final_fig)

        print(f"Training complete. Loss curve saved to {final_path}")
        print(f"CSV log saved to {csv_path}")

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
                    logits = classifier(x_t, t)
                    loss = F.cross_entropy(logits, y)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                self.opt_classifier.step()
                self.opt_classifier.zero_grad()
                self.step_classifier += 1
                step += 1

                with torch.no_grad():
                    if self.step_classifier != 0 and self.step_classifier % self.save_cycle == 0:
                        self.milestone_classifier += 1
                        self.save_classifier(self.milestone_classifier)
                                            
                    if self.logger is not None and self.step_classifier % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

        # return classifier

