import numpy as np
import glob
import os
import json
import pickle
# from collections import defaultdict

if __name__ == '__main__':
    seq_dict_path = '/Users/bismarck/Downloads/temp_data/seq_dict.pkl'
    text_proc_path = '/Users/bismarck/Downloads/temp_data/torchtext.pkl'
    
    with open(seq_dict_path, 'rb') as f:
        seq_dict = pickle.load(f)
    with open(text_proc_path, 'rb') as f:
        text_proc = pickle.load(f)

    print(type(seq_dict))
    print(type(text_proc))

    print('size of seq_dict is {siz}'.format(siz=len(seq_dict)))
    print('size of text_proc.vocab is {siz}'.format(siz=len(text_proc.vocab.stoi)))
    
    for key in seq_dict.keys():
        print(seq_dict[key])
        break
