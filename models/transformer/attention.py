#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    """
    Scaled Dot Product Attention.
    """
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        # Using mask to constrain the attention range.
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.3):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, 'd_model can not be divided by n_heads!'

        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        residual = query.detach().clone()

        query, key, value = [l(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linear_layers, (query, key, value))]
        

        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        x = self.dropout(self.output_linear(x))
        x += residual
        x = self.layer_norm(x)

        return x, attn

