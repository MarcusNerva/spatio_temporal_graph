import numpy as np
import torch
import torch.nn as nn
import glob
import os
import json
import pickle
import sys
sys.path.append('../')

if __name__ == '__main__':
    from dataset import DatasetMSRVTT
    from cfgs import get_total_settings
    from torch.utils.data import DataLoader

    args = get_total_settings()
    valid_set = DatasetMSRVTT('valid', args)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)

    for i, (G_st, F_O, resnet_2d, i3d_3d, sentences) in tqdm(enumerate(test_loader)):
        print(sentences)
        break
