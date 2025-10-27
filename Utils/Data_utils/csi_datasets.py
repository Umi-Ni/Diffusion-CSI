import os
import torch
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    """
    通用时序数据集封装：负责数据读取、归一化/反归一化、滑窗切片、训练/测试划分，以及（测试期）缺失掩码生成。

    意图：
    - 读取 CSV（或子类自定义读取）并拟合 MinMaxScaler；可选将数据映射到 [-1, 1] 区间。
    - 按 window 进行等长滑窗切片，再按比例划分 train/test。
    - 测试期支持两种评估模式：随机缺失（missing_ratio）或固定尾部预测（predict_length）。

    形状约定：
    - 原始原型 rawdata: (N_rows, N_features)
    - 切片 samples: (N_samples, window, N_features)
    - __getitem__:
      - train -> Tensor(float32): (window, N_features)
      - test  -> Tuple[Tensor(float32) (window, N_features), BoolTensor (window, N_features)]
    """
    def __init__(
        self, 
        name: str,
        data_root: str,
        window: int = 500,
        proportion: float = 0.8,
        save2npy: bool = True,
        neg_one_to_one: bool = True,
        seed: int = 123,
        period: str = 'train',
        output_dir: str = './OUTPUT',
        predict_length: Optional[int] = None,
        missing_ratio: Optional[float] = None,
        style: str = 'separate', 
        distribution: str = 'geometric', 
        mean_mask_length: int = 3
    ) -> None:
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            # 训练期不应设置预测长度或缺失率（修复：避免使用按位取反 ~ 导致断言失效）
            assert (predict_length is None and missing_ratio is None), \
                'train 期不应设置 predict_length 或 missing_ratio'
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        # 命名优化：避免覆盖内置 len；使用更具语义的名称
        self.n_rows, self.n_features = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.n_segments_total = self.n_rows // self.window
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.data = self.__normalize(self.rawdata)
        train, inference = self._get_samples(self.data, proportion, seed)

        self.samples = train if period == 'train' else inference
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.n_samples = self.samples.shape[0]

    def _get_samples(self, data: np.ndarray, proportion: float, seed: int):
        """
        将连续数据切成 (num_segments, window, n_features) 并按比例划分训练/测试。
        注意：当前实现直接 reshape，若总长度不能被 window 整除会潜在报错。
        TODO: 可裁剪到 usable_len = (len // window) * window 再 reshape，以避免越界。
        """
        num_segments = data.shape[0] // self.window
        # x: (num_segments, window, n_features)
        x = data.reshape(num_segments, self.window, self.n_features)

        train_data, test_data = self.divide(x, proportion, seed)

        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

        return train_data, test_data

    def normalize(self, sq):
        # sq: (batch*?, window, n_features) or (N, n_features)
        d = sq.reshape(-1, self.n_features)  # (N_total, n_features)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.n_features)

    def unnormalize(self, sq):
        # sq: (batch*?, window, n_features)
        d = self.__unnormalize(sq.reshape(-1, self.n_features))  # (N_total, n_features)
        return d.reshape(-1, self.window, self.n_features)
    
    def __normalize(self, rawdata):
        # rawdata: (N_rows, n_features)
        data = self.scaler.transform(rawdata)  # (N_rows, n_features)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        # data: (N, n_features)
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    @staticmethod
    def divide(data, ratio, seed=2023):
        """
        按比例切分数据（当前实现为稳定顺序切分，而非随机打乱）。
        data: (num_segments, window, n_features) -> (train, test)
        """
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        # id_rdm = np.random.permutation(size)
        id_rdm = np.arange(size)
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
        - 返回 (numpy array, fitted MinMaxScaler)
        """
        # 先用 header=None 读取，检查首行是否为非数值（即可能为列名）
        try:
            df_try = pd.read_csv(filepath, header=None)
        except Exception as e:
            raise RuntimeError(f"读取 CSV 失败: {filepath}\n{e}")

        # 检查首行是否包含非数值内容（判断为 header 的可能性）
        first_row = df_try.iloc[0].astype(str)
        is_header = True
        for v in first_row:
            # 认为纯数字（含小数/负号）为数值，否则视为 header 字符串
            s = v.strip()
            if s == '':
                # 空字符串也认为非 header（保守）
                is_header = True
                break
            # 尝试把字符串转换为 float 判断
            try:
                float(s)
                # 如果能成功转换为 float，则不是 header -> 继续判断下一个
                # （但遇到若干列均为数值则仍可能没有 header）
                continue
            except:
                # 无法转为 float，则很可能首行为 header（列名）
                is_header = True
                break
        else:
            # 如果循环没有 break（即所有首行都能转成数值），则首行很可能不是 header
            is_header = False

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

        data = df.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        # 可选：打印诊断信息，便于调试
        print(f"[read_data] {os.path.basename(filepath)} -> shape: {data.shape}; columns: {df.shape[1]}")
        return data, scaler

    
    def mask_data(self, seed=2023):
        """
        生成随机缺失掩码（测试期使用）。
        返回 masks: bool array with shape (N_samples, window, n_features)
        """
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
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array -> (window, n_features)
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array -> (window, n_features)
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array -> (window, n_features)
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.n_samples
    

class fMRIDataset(CustomDataset):
    """
    fMRI 数据集示例：覆写 read_data，从 Mat 文件读取时序矩阵。

    形状：data: (N_rows, N_features)
    其余流程沿用父类（归一化、切片、划分等）。
    """
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
