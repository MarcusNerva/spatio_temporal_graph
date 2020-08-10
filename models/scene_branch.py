#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from transformer import Transformer
from encoders import SceneEncoder

class SceneBranch(nn.Module):
    """
    SceneBranch = SceneEncoder(encoder) + Transformer(decoder)
    """
    def __init__(self, T, d_2D, d_3D, out_feature_size, n_vocab, pad_idx, 
                enc_drop_prob=0.5, trans_drop_prob=0.3,
                 n_layers=2, n_heads=8, d_model=512, d_hidden=1024, emb_prj_weight_sharing=True):
        """
        Args:
            T: the number of features' frame.
            d_2D: the length of 2D features' dim.
            d_3D: the length of 3D features' dim.
            out_feature_size: the length of encoder's output dim.
        """
        super(SceneBranch, self).__init__()

        self.encoder = SceneEncoder(T=T, d_2D=d_2D, d_3D=d_3D, d_model=out_feature_size, drop_probability=enc_drop_prob)
        self.decoder = Transformer(n_vocab=n_vocab, pad_idx=pad_idx, n_layers=n_layers, n_heads=n_heads,
                                  d_model=d_model, d_hidden=d_hidden, dropout=trans_drop_prob,
                                  emb_prj_weight_sharing=emb_prj_weight_sharing)

    def forward(self, F_2D, F_3D, gt_seq):
        """
        Args:
            F_2D: 2d visual features.
            F_3D: 3d visual features.
            gt_seq: ground truth sentences.

        Shapes:
            F_2D: (batch, T, d_2D)
            F_3D: (batch, T, d_3D)
            gt_seq: (batch, max_seq_len)
            ----------------------------------
            distribution: (batch * max_seq_len, n_vocab)

        Returns:
            distribution: the probability distribution across vocabulary V.

        """
        visual_features = self.encoder(F_2D=F_2D, F_3D=F_3D)
        distribution = self.decoder(features=visual_features, gt_seq=gt_seq)
        return distribution;

    def generate_sentence(self, F_2D, F_3D, beam_size, max_seq_len, bos_idx, eos_idx):
        """
        generate sentences when testing.
        """

        batch = F_2D.shape[0]

        visual_features = self.encoder(F_2D=F_2D, F_3D=F_3D)
        visual_features = list(visual_features.split(split_size=1, dim=0))
        
        ret = []
        for i in range(batch):
            ret.append(self.decoder.generate_sentence(features=visual_features[i], beam_size=beam_size, 
                                                      max_seq_len=max_seq_len,
                                                      bos_idx=bos_idx, eos_idx=eos_idx))
        return ret;

if __name__ == '__main__':
    batch_size = 64
    T = 10
    d_2D = 2048
    d_3D = 1024
    d_model = 512

    beam_size=5
    max_seq_len = 30
    n_vocab = 10000
    pad_idx = 9999
    
    temp_model = SceneBranch(T=T, d_2D=d_2D, d_3D=d_3D, out_feature_size=d_model, n_vocab=n_vocab, pad_idx=pad_idx)
    F_2d = torch.ones((batch_size, 10, d_2D))
    F_3d = torch.ones((batch_size, 10, d_3D))
    gt_seq = torch.ones((batch_size, max_seq_len)).long()

    out = temp_model(F_2d, F_3d, gt_seq)
    print("out.shape == ", out.shape)

    seqs = temp_model.generate_sentence(F_2D=F_2d, F_3D=F_3d, beam_size=beam_size, max_seq_len=max_seq_len, bos_idx=0, eos_idx=9998)
    for i in range(batch_size):
        print(seqs[i].shape)
