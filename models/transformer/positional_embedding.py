#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, n_position=512):
        super(PositionalEmbedding, self).__init__()
        
        pe = torch.zeros(n_position, d_model)
        pe.requires_grad = False

        position = torch.arange(0, n_position).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
    
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].clone().detach()


