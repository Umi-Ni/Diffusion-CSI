import torch
import torch.nn.functional as F
from torch import nn


class FreqAttention(nn.Module):
    """
    Frequency-axis multi-head self-attention.

    - Operates on input x with shape (B, T, C) and computes attention along the
      frequency/subcarrier axis (F). We assume that the model knows F (feature_size)
      and will project embedding into F tokens per time step.

    Design notes:
    - Input x -> linear projection to produce per-frequency Q/K/V tokens: (B, T, F, n_head*head_dim)
    - Attention computed along F dimension for each (B, T, head).
    - Learnable frequency positional embedding is added to tokens before Q/K.
    - Returns an output of shape (B, T, C) to be merged as a residual branch.

    Shapes (internal):
    - x: (B, T, C)
    - after proj: (B, T, F, total_head_dim)
    - att: (B, n_head, T, F, F)

    """

    def __init__(
        self,
        n_embd: int,
        freq_size: int,
        n_head: int = 4,
        head_dim: int = 16,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
    ):
        super().__init__()
        assert n_embd % 1 == 0
        self.n_head = n_head
        self.head_dim = head_dim
        self.freq_size = freq_size
        self.total_head_dim = n_head * head_dim

        # project from model dim to per-frequency multi-head dims
        self.to_q = nn.Linear(n_embd, freq_size * self.total_head_dim)
        self.to_k = nn.Linear(n_embd, freq_size * self.total_head_dim)
        self.to_v = nn.Linear(n_embd, freq_size * self.total_head_dim)

        # frequency positional embedding (learnable)
        self.freq_pos = nn.Parameter(torch.zeros(1, freq_size, self.total_head_dim))
        nn.init.normal_(self.freq_pos, std=0.02)

        # output projection
        self.proj = nn.Linear(freq_size * self.total_head_dim, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

    def forward(self, x):
        """
        x: (B, T, C)
        returns: (out: (B, T, C), attn_mean: (B, T, F, F) averaged over heads)
        """
        B, T, C = x.shape

        # Project to per-frequency Q/K/V and reshape
        # q/k/v: (B, T, F, total_head_dim)
        q = self.to_q(x).view(B, T, self.freq_size, self.total_head_dim)
        k = self.to_k(x).view(B, T, self.freq_size, self.total_head_dim)
        v = self.to_v(x).view(B, T, self.freq_size, self.total_head_dim)

        # add learnable frequency positional embedding
        q = q + self.freq_pos  # broadcast on batch/time
        k = k + self.freq_pos

        # reshape for multi-head: (B, n_head, T, F, head_dim)
        q = q.view(B, T, self.freq_size, self.n_head, self.head_dim).permute(0, 3, 1, 2, 4)
        k = k.view(B, T, self.freq_size, self.n_head, self.head_dim).permute(0, 3, 1, 2, 4)
        v = v.view(B, T, self.freq_size, self.n_head, self.head_dim).permute(0, 3, 1, 2, 4)

        # attention along frequency axis F (dim -2 vs -1 after permute)
        # q,k: (B, n_head, T, F, head_dim)
        scale = 1.0 / (self.head_dim ** 0.5)
        att = torch.einsum('bhtfd,bhtgd->bhtfg', q, k) * scale  # (B, n_head, T, F, F)

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # aggregate
        out = torch.einsum('bhtfg,bhtgd->bhtfd', att, v)  # (B, n_head, T, F, head_dim)

        # merge heads and freq tokens back
        out = out.permute(0, 2, 3, 1, 4).contiguous().view(B, T, self.freq_size * self.total_head_dim)

        out = self.resid_drop(self.proj(out))  # (B, T, C)

        # average attention across heads for optional diagnostics: (B, T, F, F)
        att_mean = att.mean(dim=1)
        return out, att_mean
