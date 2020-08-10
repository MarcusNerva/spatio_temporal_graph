#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('../')
import torch
import torch.nn as nn
from transformer.attention import MultiHeadAttention
from transformer.positionwise_feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    EncoderLayer = self_attention with sublayerconnection + feedforward with sublayerconnection
    """
    def __init__(self, d_model, d_hidden, n_heads, dropout=0.3):
        """
        Args:
            d_model: hidden size of transformer
            d_hidden: feed forward hidden size, usually 4 * d_model
            n_heads: number of heads in multi-head attention
            dropout: dropout rate
        """
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)
        self.pos_feedforward = PositionwiseFeedForward(d_model=d_model, d_hidden=d_hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, encoder_input, mask=None):
        encoder_input, encoder_attn = self.self_attention(encoder_input, encoder_input, encoder_input, mask=mask)
        encoder_output = self.pos_feedforward(encoder_input)
        return encoder_output, encoder_attn
        

class DecoderLayer(nn.Module):
    """
    DecoderLayer = self_attention + enc_attention + feedforward
    """
    def __init__(self, d_model, d_hidden, n_heads, dropout=0.3):
        """
        Args:
            d_model: hidden size of transformer
            d_hidden: feed forward hidden size, usually 4 * d_model
            n_heads: number of heads in multi-head attention
            dropout: dropout rate
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)
        self.enc_attention = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)
        self.pos_feedforward = PositionwiseFeedForward(d_model=d_model, d_hidden=d_hidden, dropout=dropout)

    def forward(self, encoder_output, decoder_input, self_attn_mask=None, enc_attn_mask=None):
        decoder_output, dec_self_attn = self.self_attention(decoder_input, decoder_input, decoder_input, mask=self_attn_mask)
        decoder_output, dec_enc_attn = self.enc_attention(decoder_output, encoder_output, encoder_output, mask=enc_attn_mask)
        decoder_output = self.pos_feedforward(decoder_output)
        return decoder_output, dec_self_attn, dec_enc_attn

if __name__ == '__main__':
    d_model = 512
    d_hidden = 2048
    n_heads = 8
    batch_size = 16
    len_frames = 20

    encoder_input = torch.ones(batch_size, len_frames, d_model)
    decoder_output = torch.ones(batch_size, len_frames // 2, d_model)
    encoder_mask = torch.ones(batch_size, 1, len_frames)
    self_attn_mask = (1 - torch.triu(torch.ones(1, len_frames // 2, len_frames // 2), diagonal=1))
    dec_attn_mask = encoder_mask.clone()
    
    temp_encoder = EncoderLayer(d_model=d_model, d_hidden=d_hidden, n_heads=n_heads)
    temp_decoder = DecoderLayer(d_model=d_model, d_hidden=d_hidden, n_heads=n_heads)

    encoder_output = temp_encoder(encoder_input=encoder_input, mask=encoder_mask)
    print(encoder_output.shape)
    decoder_output, *_ = temp_decoder(encoder_output=encoder_output, decoder_input=decoder_output,
                                      self_attn_mask=self_attn_mask, dec_attn_mask=dec_attn_mask)
    print(decoder_output.shape)


