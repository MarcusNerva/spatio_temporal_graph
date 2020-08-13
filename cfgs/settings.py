#!/usr/bin/env python
# coding=utf-8
import argparse

def get_total_settings():
    parser = argparse.ArgumentParser()

    # data path setting
    # parser.add_argument('--data_path', type=str, default='/Users/bismarck/Downloads/temp_data', help='the path of all kinds of data')
    parser.add_argument('--data_path', type=str, default='/disks/lilaoshi666/hanhua.ye/spatio_temporal_graph/data', help='the path of all kinds of data')

    # object branch encoder setting
    parser.add_argument('--object_in_feature_size', type=int, default=1024, help='feature size of object branch input features')
    parser.add_argument('--object_out_feature_size', type=int, default=512, help='feature size of object branch output features')
    parser.add_argument('--object_N', type=int, default=50, help='the total number of object in a single video')

    # scene branch encoder setting
    parser.add_argument('--scene_T', type=int, default=10, help='the number of features\' frames per video')
    parser.add_argument('--scene_d_2D', type=int, default=2048, help='the size of 2D features input to encoder part')
    parser.add_argument('--scene_d_3D', type=int, default=1024, help='the size of 3D features input to encoder part')
    parser.add_argument('--scene_output_features', type=int, default=512, help='the size of output features of encoder part, namely, the size of input features of decoder')
    parser.add_argument('--beam_size', type=int, default=5, help='the size of beam searching')
    parser.add_argument('--max_seq_len', type=int, default=20, help='the max length of sequence')

    # the common settings for both object branch and scene branch
    parser.add_argument('--encoder_drop', type=float, default=0.5, help='the drop probability of both object and scene encoders')

    # decoder part setting, namely, transformer
    parser.add_argument('--n_layers', type=int, default=2, help='the number of layers in transformer\'s encoder and decoder block')
    parser.add_argument('--n_heads', type=int, default=8, help='the number of head in multihead attention')
    parser.add_argument('--d_model', type=int, default=512, help='the input and output size of transformer')
    parser.add_argument('--d_hidden', type=int, default=1024, help='the hidden size of transformer')
    parser.add_argument('--emb_prj_weight_sharing', action='store_true', help='whether share the weight between target word embedding & last dense layer or not')
    parser.add_argument('--trans_drop', type=float, default=0.3, help='the dropout probability of transformer')
    parser.add_argument('--n_vocab', type=int, help='the number of words\' kinds in caption process')
    parser.add_argument('--pad', type=str, default='<pad>', help='the pad_idx of sentences')
    parser.add_argument('--bos', type=str, default='<bos>', help='the signal of sentence beginning')
    parser.add_argument('--eos', type=str, default='<eos>', help='the signal of sentence endding')

    # the hyperparameter of training
    
    args = parser.parse_args()
    return args
