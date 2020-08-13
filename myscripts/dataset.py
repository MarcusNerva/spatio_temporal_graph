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
            spatial_matrix_path = os.path.join(object_object_path, video_id + '.npy')
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
            sentencec_proc: a list contains 20 element, each of which is a list of words/numbers.
        """

        sentencec_proc = list(map(text_proc.preprocess, sentences))
        if self.mode in ['valid', 'test']:
            return sentencec_proc
        sentencec_proc = text_proc.numericalize(text_proc.pad(sentences_proc))
        return sentencec_proc

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
    from cfgs import get_total_settings

    args = get_total_settings()
    get_vocab_and_seq(args)
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

