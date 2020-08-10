#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from transformer import Transformer
from encoders import GCN

class ObjectBranch(nn.Module):
    """
    ObjectBranch = Graph convolution network(encoder) + Transformer(decoder)
    """
    def __init__(self, in_feature_size, out_feature_size, N, 
                 n_vocab, pad_idx, 
                 enc_drop_prob=0.5, dec_drop_prob=0.3,
                 n_layers=2, n_heads=8, d_model=512, d_hidden=1024, emb_prj_weight_sharing=True):
        """
        Args:
            in_feature_size: feature_size of input features, namely, F_O.
            out_feature_size: feature_size of output features, namely, visual features utilized by decoder.
            N: sum of object number in the extracted frames.
            n_vocab: the size of vocabulary V.
            pad_idx: the pad_idx in generated sentences.

        """
        super(ObjectBranch, self).__init__()
        
        self.encoder = GCN(in_feature_size=in_feature_size, out_feature_size=out_feature_size, 
                           N=N, drop_probability=enc_drop_prob)

        self.decoder = Transformer(n_vocab=n_vocab, pad_idx=pad_idx, n_layers=n_layers, n_heads=n_heads,
                                  d_model=d_model, d_hidden=d_hidden, dropout=dec_drop_prob, 
                                  emb_prj_weight_sharing=emb_prj_weight_sharing)

    def forward(self, G_st, F_O, gt_seq):
        """
        Args:
            G_st: spatio-temporal graph.
            F_O: Object features extracted by Faster R-CNN.
            gt_seq: ground truth sentences.

        Shapes:
            G_st: (batch, N, N)
            F_O: (batch, T, d_model)
            gt_seq: (batch, max_seq_len)
            -------------------------------
            distribution: (batch * max_seq_len, n_vocab)

        Returns:
            distribution: the probability distribution across vocabulary V.

        """
        visual_features = self.encoder(G_st, F_O)
        distribution = self.decoder(features=visual_features, gt_seq=gt_seq)
        return distribution

if __name__ == '__main__':
    batch_size = 64
    N = 50
    d_2d = 1024
    d_model = 512
    n_vocab = 10000
    pad_idx = 9999
    max_seq_len = 30

    G_st = torch.ones((batch_size, N, N))
    F_O = torch.ones((batch_size, N, d_2d))
    gt_seq = torch.ones((batch_size, max_seq_len)).long()
    temp_model = ObjectBranch(in_feature_size=d_2d, out_feature_size=d_model, N=N, n_vocab=n_vocab, pad_idx=pad_idx)

    out = temp_model(G_st=G_st, F_O=F_O, gt_seq=gt_seq)
    print(out.shape)

