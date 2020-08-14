import numpy as np
import torch
import torch.nn as nn
import glob
import os
import json
import pickle

class Experiment(nn.Module):
    def __init__(self):
        super(Experiment, self).__init__()
        self.register_buffer('blank_seq', torch.ones((3, 3)))

    def generate(self):
        self.register_buffer('full_seq', torch.zeros((3, 3)))

    def get_device_origin(self):
        return self.blank_seq.device

    def get_device_generate(self):
        self.generate()
        return self.full_seq.device

if __name__ == '__main__':

    exper = Experiment()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exper.to(device)

    print(exper.get_device_origin())
    print(exper.get_device_generate())

