import math
import torch
import torch.nn.functional as F
from torch import nn
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

class FullAttention(nn.Module):
    """
    标准多头自注意力（可选 RoPE 旋转位置编码）。

    参数：
    - n_embd: 嵌入维度 C
    - n_head: 头数，需满足 n_embd % n_head == 0
    - attn_pdrop/resid_pdrop: 注意力/残差的 dropout 比例
    - use_rope: 是否启用 RoPE 位置编码
    - max_seq_len: RoPE 的最大序列长度（如底层实现需要）

    输入/输出：
    - x: (B, T, C)
    - 返回: (y: (B, T, C), attn_mean: (B, T, T))  # 对所有头取平均
    """
    def __init__(self,
                 n_embd,          # embedding dimension
                 n_head,          # number of attention heads
                 attn_pdrop=0.1,  # attention dropout
                 resid_pdrop=0.1, # residual dropout
                 use_rope=False,  # 是否启用RoPE位置编码
                 max_seq_len=512  # RoPE最大序列长度（可调）
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.use_rope = use_rope

        # QKV投影
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # 正则化与输出
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)

        # 若启用RoPE，初始化旋转位置编码
        if self.use_rope:
            head_dim = n_embd // n_head
            self.rotary_emb = RotaryEmbedding(dim=head_dim)

    def forward(self, x, mask=None):
        """
        自注意力前向：可选 RoPE，对 (B,T,C) 序列进行多头点乘注意力。
        Args:
            x: Tensor (B, T, C)
            mask: Optional[Tensor], 支持 (B, T) 或已广播到 (B,1,1,T)
        Returns:
            y: (B, T, C), attn_mean: (B, T, T)
        """
        B, T, C = x.size()

        # Q, K, V
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)    # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # RoPE
        if self.use_rope:
            pos = torch.arange(T, device=x.device)
            rotary_pos_emb = self.rotary_emb(pos)
            q, k = apply_rotary_emb(rotary_pos_emb, q, k)

        # 点乘注意力
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # (B, n_head, T, T)
        if mask is not None:
            # 自动调整mask形状
            # att.shape: (B, n_head, T, T)
            # mask 支持 (B, T) 或 (B, 1, 1, T)
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            elif mask.dim() == 3 and mask.shape[1] == 1:
                mask = mask[:, :, None, :]
            # 裁剪/填充到 T
            T_att = att.size(-1)
            T_mask = mask.size(-1)
            if T_mask < T_att:
                pad_len = T_att - T_mask
                mask = F.pad(mask, (0, pad_len), value=1)
            elif T_mask > T_att:
                mask = mask[..., :T_att]
            att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # 聚合输出
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # -> (B, T, C)
        y = self.resid_drop(self.proj(y))

        # 平均注意力图（便于可视化）
        att_mean = att.mean(dim=1, keepdim=False)
        return y, att_mean
