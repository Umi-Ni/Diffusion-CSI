import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class RobustScaler1D:
    """Robust per-feature scaler for log-amplitude CSI.

    - Fit on training split only, per subcarrier (feature).
    - Center: median.
    - Scale: IQR (Q75 - Q25). If IQR is too small, fallback to 1.4826 * MAD; if still too small, fallback to std; else use 1.0.
    - Supports 2D shape [N, D] and 3D shape [N, T, D].
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.median = None  # [D]
        self.scale = None   # [D]

    def _ensure_2d(self, x: np.ndarray):
        if x.ndim == 2:
            return x, None
        elif x.ndim == 3:
            n, t, d = x.shape
            return x.reshape(n * t, d), (n, t, d)
        else:
            raise ValueError(f"Unsupported shape for RobustScaler1D: {x.shape}")

    def fit(self, x: np.ndarray):
        x2, _ = self._ensure_2d(x)
        # median
        med = np.median(x2, axis=0)
        # IQR
        q25 = np.percentile(x2, 25.0, axis=0)
        q75 = np.percentile(x2, 75.0, axis=0)
        iqr = q75 - q25
        # MAD (Median Absolute Deviation)
        mad = np.median(np.abs(x2 - med), axis=0)
        mad_sigma = 1.4826 * mad
        # STD fallback
        std = np.std(x2, axis=0)

        scale = iqr.copy()
        # Fallbacks where IQR too small
        too_small = scale < self.eps
        scale[too_small] = mad_sigma[too_small]
        too_small = scale < self.eps
        scale[too_small] = std[too_small]
        too_small = scale < self.eps
        scale[too_small] = 1.0

        self.median = med.astype(np.float64)
        self.scale = scale.astype(np.float64)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x2, shp = self._ensure_2d(x)
        z = (x2 - self.median) / (self.scale + self.eps)
        if shp is None:
            return z
        return z.reshape(shp)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        z2, shp = self._ensure_2d(z)
        x = z2 * (self.scale + self.eps) + self.median
        if shp is None:
            return x
        return x.reshape(shp)


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        window=500, 
        proportion=0.8, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            # For training period, predict_length and missing_ratio should not be set
            assert predict_length is None and missing_ratio is None, 'predict_length and missing_ratio must be None for training period'
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        # 读取并仅做对数幅度变换（不做MinMax），返回log域数据
        self.rawdata_log = self.read_data(data_root, self.name)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata_log.shape[0], self.rawdata_log.shape[-1]
        self.sample_num_total = self.len // self.window
        self.save2npy = save2npy
        # auto_norm=True 表示在robust标准化后，再进行[-k,k] -> [-1,1]的线性缩放（可逆），以便与下游保持接口一致。
        self.auto_norm = neg_one_to_one
        self.robust_clip = 3.0  # k：线性缩放范围，常用3~5

        # 先按窗口切分并划分train/test，再基于train拟合RobustScaler（避免数据泄漏）
        train_raw, test_raw = self.__getsamples_raw(self.rawdata_log, proportion, seed)
        # 拟合RobustScaler（在log域上）
        self.scaler = RobustScaler1D().fit(train_raw)
        # 变换到robust z-score域
        train_z = self.scaler.transform(train_raw)
        test_z = self.scaler.transform(test_raw)
        # 若需要与模型接口保持[-1,1]的范围，可做线性缩放：clip(z, -k, k)/k
        if self.auto_norm:
            train_norm = np.clip(train_z, -self.robust_clip, self.robust_clip) / self.robust_clip
            test_norm = np.clip(test_z, -self.robust_clip, self.robust_clip) / self.robust_clip
        else:
            train_norm, test_norm = train_z, test_z

        # 保存样本（规范为 [num_segments, window, var_num]）
        self.samples = train_norm if period == 'train' else test_norm
        # 可选：保存规范化后的数据（便于对照与复现实验）
        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_norm)
            np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_norm)
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]

    def __getsamples_raw(self, data_log: np.ndarray, proportion, seed):
        """将log域原始数据按窗口切分->划分train/test。
        返回：train_raw, test_raw，形状均为 [num_segments, window, var_num]
        """
        num_segments = data_log.shape[0] // self.window
        x = data_log[: num_segments * self.window].reshape(num_segments, self.window, self.var_num)

        train_data, test_data = self.divide(x, proportion, seed)

        # 立即保存未标准化的线性幅度ground truth供可视化对比
        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize_from_log_domain(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize_from_log_domain(train_data))

        return train_data, test_data

    def normalize(self, sq):
        """对输入做robust标准化（log域->z-score），并可选做[-1,1]缩放。保留兼容性供外部调用。"""
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = np.clip(d, -self.robust_clip, self.robust_clip) / self.robust_clip
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        """从模型域还原到线性幅度域。
        输入sq形状为 [num_segments, window, var_num]，其值可能在[-1,1]（auto_norm=True）或为robust z-score（auto_norm=False）。
        """
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        """保持旧接口（未使用）。建议使用 normalize() 并先fit scaler在train上。"""
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = np.clip(data, -self.robust_clip, self.robust_clip) / self.robust_clip
        return data

    def __unnormalize(self, data):
        """将模型输出还原为线性幅度。
        - 若auto_norm=True，先把[-1,1]缩放回[-k,k]的robust z-score域。
        - 再用RobustScaler逆变换回log域。
        - 最后exp回到线性幅度域。
        """
        if self.auto_norm:
            data = np.clip(data, -1.0, 1.0) * self.robust_clip
        # z-score -> log域
        x_log = self.scaler.inverse_transform(data)
        # log域 -> 线性幅度
        x = np.expm1(x_log)
        return x

    def unnormalize_from_log_domain(self, data_log_seq):
        """辅助函数：已在log域的序列（未做标准化），直接exp回线性幅度域。
        输入/输出形状与 data_log_seq 相同。
        """
        return np.expm1(data_log_seq)
    
    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        # use a random permutation for splitting (deterministic given seed)
        id_rdm = np.random.permutation(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv robustly.
        - 自动检测是否存在 header（如 'sub01,sub02,...'）
        - 删除 pandas 自动产生的 Unnamed 列（通常是保存时带 index 导致）
        - 返回 log1p(幅度) 的 numpy 数组（不做任何基于全数据的缩放，避免数据泄漏）
        """
        # 先用 header=None 读取，检查首行是否为非数值（即可能为列名）
        try:
            df_try = pd.read_csv(filepath, header=None)
        except Exception as e:
            raise RuntimeError(f"读取 CSV 失败: {filepath}\n{e}")

        # 检查首行是否包含非数值内容（判断为 header 的可能性）
        first_row = df_try.iloc[0].astype(str)
        # assume no header; if any value in first row is non-numeric or empty, treat as header
        is_header = False
        for v in first_row:
            s = v.strip()
            if s == '':
                is_header = True
                break
            try:
                float(s)
            except Exception:
                is_header = True
                break

        # 根据检测结果选择读取方式
        if is_header:
            df = pd.read_csv(filepath, header=0)
        else:
            df = pd.read_csv(filepath, header=None)

        # 删除 pandas 可能引入的 unnamed 列（例如保存时带 index 导致的 'Unnamed: 0'）
        unnamed_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('Unnamed')]
        if len(unnamed_cols) > 0:
            df = df.drop(columns=unnamed_cols)

        # 有些 CSV 第一列可能不应该存在（比如保存时多余的索引列），如果第一列名或第一列值可疑也删除
        # 进一步检查：如果列名中第一个看起来像索引（例如 '0' 且后续列名很多），可尝试删除它（保守策略）
        if df.shape[1] > 1:
            first_col_vals = df.iloc[:, 0].astype(str)
            # 如果第一列的很多值可解析为整数且随行递增（疑似原索引），则删除
            try:
                as_ints = first_col_vals.str.match(r'^\d+$').sum()
                if as_ints / float(len(first_col_vals)) > 0.9:
                    # 90% 以上为纯整数字符串 -> 很可能是索引列
                    df = df.drop(df.columns[0], axis=1)
            except Exception:
                pass

        # 最后做一次类型转换，确保数据为数值
        df = df.apply(pd.to_numeric, errors='coerce')

        # 如果存在 NaN，给出提示（可能说明 header 没读对）
        if df.isnull().values.any():
            nan_count = df.isnull().sum().sum()
            print(f"⚠️ 注意: 读取后发现 {nan_count} 个 NaN，可能是 header/格式问题。文件: {filepath}")

        data = df.values.astype(np.float64)
        # 对幅度做对数变换以缓和长尾与乘性噪声（仅此一步，不做全数据缩放）
        data_log = np.log1p(np.clip(data, a_min=0.0, a_max=None))
        # 可选：打印诊断信息，便于调试
        print(f"[read_data] {os.path.basename(filepath)} -> shape: {data.shape}; columns: {df.shape[1]} | using log1p only (no global scaler)")
        return data_log

    
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
    

class fMRIDataset(CustomDataset):
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Read fMRI dataset; keep as raw array (no global scaler, follow robust pipeline)."""
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        return data.astype(np.float64)
