import numpy as np
import torch
import glob
import os
import json
import pickle
np.set_printoptions(threshold=np.inf)
# from collections import defaultdict

def _construct_G_st(spatial_matrix, temporal_matrix):

    spatial_matrix = torch.from_numpy(spatial_matrix)
    temporal_matrix = torch.from_numpy(temporal_matrix)

    N = 50
    G_st = torch.zeros((N, N))
    for i, j in zip(range(10), range(0, N, 5)):
        G_st[j : j + 5, j : j + 5] = spatial_matrix[i]
    for i, j in zip(range(9), range(5, N, 5)):
        G_st[j - 5 : j, j : j + 5] = temporal_matrix[i]

    return G_st.cpu().numpy().astype(np.int32)

if __name__ == '__main__':
    spatial_matrix = np.ones((10, 5, 5))
    temporal_matrix = np.ones((9, 5, 5)) * 2

    G_st = _construct_G_st(spatial_matrix, temporal_matrix)
    for item in G_st:
        temp = str()
        for idx in item:
            temp += str(idx);
        print(temp)
