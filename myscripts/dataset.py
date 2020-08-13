#!/usr/bin/env python
# coding=utf-8
import torch
import torchtext
from torch.utils.data import Dataset
import numpy as np
import os
import json
import pickle
from collections import defaultdict

def get_vocab_and_seq(args):
    """
    This function is aimming to process captions in json file. And save these data on disk through serialization.

    Args:
        args: the global settings which are defined in tools/settings.py

    Savings:
        text_proc: torchtext.data.Field instance. Return it for utilizing it's vocab
        seq_dict: {video_id -> [caption * 20]}
    """

    data_path = args.data_path
    bos = args.bos
    eos = args.eos
    pad = args.pad
    max_length = args.max_seq_len
    json_path = os.path.join(data_path, 'videodatainfo_2017.json')
    torchtext_path = os.path.join(data_path, 'torchtext.pkl')
    seq_dict_path = os.path.join(data_path, 'seq_dict.pkl')
    seq_dict = defaultdict(list)

    text_proc = torchtext.data.Field(sequential=True, init_token=pad, eos_token=eos, tokenize='spacy', lower=True,
                                    batch_first=True, fix_length=max_length)
    seqs_store = []

    with open(json_path, 'r') as f:
        json_file = json.load(f)
        sentences = json_file['sentences']
        for temp_dict in sentences:
            video_id = temp_dict['video_id']
            caption = temp_dict['caption'].strip()
            seq_dict[video_id].append(caption)
            seqs_store.append(caption)

    seqs_store = list(map(text_proc.preprocess, seqs_store))
    text_proc.build_vocab(seqs_store, min_freq=1)
    
    with open(torchtext_path, 'wb') as f:
        pickle.dump(text_proc, f)
    with open(seq_dict_path, 'wb') as f:
        pickle.dump(seq_dict, f)

class DatasetMSRVTT(Dataset):
    """
    This dataset class comprises 3 kinds of mode, namely, training, validation, testing.
    
    I utilize first 6513 videos as train_dataset, that is, video_id in [0, 6513).
    Subsequently, 497 videos in the middle is utilized as valid_dataset, that is, video_id in [6513, 7010).
    Finally, the rest videos is utilized as test_dataset, that is, video_id in [7010, 10000).
    
    Validation mode and testing mode is similar, where the model we valid/test here needs raw captions and visual features.
    In training mode, the model needs numericalized captions and visual features, however.
    """

    def __init__(self, mode, args):
        """
        Args:
            mode: 'train', 'valid', 'test'.
            args: the global settings defined in tools/settings.py
        """
        super(DatasetMSRVTT, self).__init__()
        self.mode = mode
        self.args = args
        self.N = args.object_N
        self.data_range = self._define_data_range()
        self.G_st = []
        self.F_O = []
        self.resnet_2d_feats = []
        self.i3d_3d_feats = []
        self.sentences = []
        self.pad_idx = None
        self.n_vocab = None

        data_path = args.data_path
        # visual features
        object_spatial_path = os.path.join(data_path, 'object/spatial')
        object_temporal_path = os.path.join(data_path, 'object/temporal')
        object_object_path = os.path.join(data_path, 'object/object')
        scene_2d_path = os.path.join(data_path, 'scene/2d')
        scene_I3D_path = os.path.join(data_path, 'scene/I3D')
        # captions related 
        seq_dict_path = os.path.join(data_path, 'seq_dict.pkl')
        torchtext_path = os.path.join(data_path, 'torchtext.pkl')
        
        # starting prepare dataset
        with open(seq_dict_path, 'rb') as f:
            seq_dict = pickle.load(f)
        with open(torchtext_path, 'rb') as f:
            text_proc = pickle.load(f)

        for video_id in self.data_range:
            spatial_matrix_path = os.path.join(object_spatial_path, video_id + '.npy')
            temporal_matrix_path = os.path.join(object_temporal_path, video_id + '.npy')
            f_o_path = os.path.join(object_object_path, video_id + '.npy')
            resnet_2d_path = os.path.join(scene_2d_path, video_id + '.npy')
            i3d_3d_path = os.path.join(scene_I3D_path, video_id + '.npy')
            
            spatial_matrix = np.load(spatial_matrix_path)
            temporal_matrix = np.load(temporal_matrix_path)
            f_o = np.load(f_o_path)
            resnet_2d = np.load(resnet_2d_path)
            i3d_3d = np.load(i3d_3d_path)

            self.G_st.append(self._construct_G_st(spatial_matrix, temporal_matrix))
            self.F_O.append(f_o)
            self.resnet_2d_feats.append(resnet_2d)
            self.i3d_3d_feats.append(i3d_3d)
            self.sentences += self._process_sentences(text_proc, seq_dict[video_id])

        pad = args.pad
        self.pad_idx = text_proc.vocab.stoi[pad]
        self.n_vocab = len(text_proc.vocab.stoi)

    def _define_data_range(self):
        
        if self.mode not in ['train', 'valid', 'test']:
            raise NotImplementedError

        ret = ['video' + str(i) for i in range(10000)]
        if self.mode == 'train':
            return ret[:6513]
        elif self.mode == 'valid':
            return ret[6513:7010]
        else:
            return ret[7010:]
    
    def _construct_G_st(self, spatial_matrix, temporal_matrix):
        """
        As paper request.
        """
        
        spatial_matrix = torch.from_numpy(spatial_matrix)
        temporal_matrix = torch.from_numpy(temporal_matrix)

        N = self.N
        G_st = torch.zeros((N, N))
        for i, j in zip(range(10), range(0, N, 5)):
            G_st[j : j + 5, j : j + 5] = spatial_matrix[i]
        for i, j in zip(range(9), range(5, N, 5)):
            G_st[j - 5 : j, j : j + 5] = temporal_matrix[i]
        
        return G_st

    def _process_sentences(self, text_proc, sentences):
        """
        This function aims to process input sentences according to dataset mode.
        In validation/testing mode, return raw video captions.
        In training mode, return numericalized video captions.

        Args:
            text_proc: the instance of torchtext.Field, used as a tool to process our sentences.
            sentences: the pending sentences list, that is, a list contains 20 captions.

        Returns:
            sentences_proc: a list contains 20 element, each of which is a list of words/numbers.
        """

        sentences_proc = list(map(text_proc.preprocess, sentences))
        if self.mode in ['valid', 'test']:
            return sentences_proc
        sentences_proc = text_proc.numericalize(text_proc.pad(sentences_proc))
        return sentences_proc

    def get_pad_idx(self):
        return self.pad_idx
    
    def get_n_vocab(self):
        return self.n_vocab

    def __len__(self):
        return 20 * len(self.data_range)

    def __getitem__(self, idx):
        return self.G_st[idx//20], \
                self.F_O[idx//20], \
                self.resnet_2d_feats[idx//20], \
                self.i3d_3d_feats[idx//20], \
                self.sentences[idx]


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    sys.path.append('../models/')
    print(sys.path)
    from cfgs import get_total_settings
    from models import ObjectBranch, SceneBranch

    args = get_total_settings()
    # get_vocab_and_seq(args)
    train_dataset = DatasetMSRVTT('train', args)
    valid_dataset = DatasetMSRVTT('valid', args)
    test_dataset = DatasetMSRVTT('test', args)

    train_range = train_dataset.data_range
    valid_range = valid_dataset.data_range
    test_range = test_dataset.data_range

    print('length of train_range is ', len(train_range))
    print('begin of train_range is {begin}, end of train_range is {end}'.format(begin=train_range[0], end=train_range[-1]))

    print('length of valid_range is ', len(valid_range))
    print('begin of valid_range is {begin}, end of valid_range is {end}'.format(begin=valid_range[0], end=valid_range[-1]))

    print('length of test_range is ', len(test_range))
    print('begin of test_range is {begin}, end of test_range is {end}'.format(begin=test_range[0], end=test_range[-1]))

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    from models import ObjectBranch, SceneBranch
    batch_size = 32
    
    in_feature_size = args.object_in_feature_size
    out_feature_size = args.object_out_feature_size
    N = args.object_N
    
    T = args.scene_T
    d_2D = args.scene_d_2D
    d_3D = args.scene_d_3D
    output_features = args.scene_output_features
    beam_size = args.beam_size
    max_seq_len = args.max_seq_len
    encoder_drop = args.encoder_drop

    n_layers = args.n_layers
    n_heads = args.n_heads
    d_model = args.d_model
    d_hidden = args.d_hidden
    trans_drop = args.trans_drop
    n_vocab = train_dataset.get_n_vocab()
    pad_idx = train_dataset.get_pad_idx()

    temp_object = ObjectBranch(in_feature_size=in_feature_size, out_feature_size=out_feature_size, N=N, n_vocab=n_vocab, pad_idx=pad_idx)
    temp_scene = SceneBranch(T=T, d_2D=d_2D, d_3D=d_3D, out_feature_size=output_features, n_vocab=n_vocab, pad_idx=pad_idx)
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    temp_object.to(device)
    temp_scene.to(device)

    for i, (G_st, F_O, resnet_2d, i3d_3d, sentences) in tqdm(train_loader):
        G_st = G_st.to(device)
        F_O = F_O.to(device)
        resnet_2d = resnet_2d.to(device)
        i3d_3d = i3d_3d.to(device)
        sentences = sentences.to(device)

        out0 = temp_object(G_st, F_O, sentences)
        out1 = temp_scene(resnet_2d, i3d_3d, sentences)

        print("out0.shape == ", out0.shape)
        print("out1.shape == ", out1.shape)

    for i, (G_st, F_O, resnet_2d, i3d_3d, sentences) in tqdm(valid_loader):
        resnet_2d = resnet_2d.to(device)
        i3d_3d = i3d_3d.to(device)
        out0 = temp_scene.generate_sentence(resnet_2d, i3d_3d)

    for i, (G_st, F_O, resnet_2d, i3d_3d, sentences) in tqdm(test_loader):
        pass
