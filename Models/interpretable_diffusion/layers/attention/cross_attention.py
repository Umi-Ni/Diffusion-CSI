import math
import torch
import torch.nn.functional as F
from torch import nn

class CrossAttention(nn.Module):
    """
    交叉注意力：以 encoder 输出作为 K/V，对解码侧序列 x 做聚合。

    输入/输出：
    - x: (B, T_dec, C)
    - encoder_output: (B, T_enc, C_cond)
    - 返回: (y: (B, T_dec, C), attn_mean: (B, T_dec, T_enc))
    """
    def __init__(self,
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()                 # x: (B, T_dec, C)
        B, T_E, _ = encoder_output.size()  # enc: (B, T_enc, C_cond)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_enc, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)              # (B, nh, T_dec, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_enc, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T_dec, T_enc)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T_dec, T_enc) x (B, nh, T_enc, hs) -> (B, nh, T_dec, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # -> (B, T_dec, C)
        att = att.mean(dim=1, keepdim=False) # (B, T_dec, T_enc)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att
