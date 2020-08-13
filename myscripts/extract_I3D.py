#!/usr/bin/env python
# coding=utf-8
import torch
import numpy as np
import h5py
import os
import sys
sys.path.append('../')
from tools import get_total_settings

if __name__ == "__main__":
    opts = get_total_settings()
    data_path = opts.data_path
    scene_path = os.path.join(data_path, 'scene')
    scene_I3D_dir = os.path.join(scene_path, 'I3D')
    scene_I3D_path = os.path.join(scene_path, 'I3D/I3D_feats.hdf5')

    object_path = os.path.join(data_path, 'object')

    I3D_feats = h5py.File(scene_I3D_path, 'r')
    vid_store = []

    for key in I3D_feats.keys():
        vid = int(key[3:])
        vid = 'video' + str(int(key[3:]) - 1)

        temp = I3D_feats[key][()]
        length = temp.shape[0]
        idxs = np.round(np.linspace(0, length, 10, endpoint=False)).astype(np.int32)
        temp = temp[idxs]
        temp_path = os.path.join(scene_I3D_dir, vid + '.npy')
        np.save(temp_path, temp)

