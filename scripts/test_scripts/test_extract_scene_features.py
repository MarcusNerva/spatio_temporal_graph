from scripts import process_frames
from scripts import extract_frames
from scripts import compute_optical_flow
from scripts import save_flows
from scripts import extract_feats
from collections import defaultdict

import torchvision
import torch
import unittest

class extractSceneFeatures(unittest.TestCase):
    def test_process_frames(self):
        frames_path = '/Users/bismarck/Downloads/temp_data/scene/frames'
        process_frames(frames_path)

    def test_extract_frames(self):
        video_path = '/Users/bismarck/Downloads/temp_data/videos/video0.mp4'
        dst_path = '/Users/bismarck/Downloads/temp_data/scene/frames'
        extract_frames(video_path, dst_path)
        pass

    def test_compute_optical_flow(self):
        pass

    def test_save_flows(self):
        pass

    def test_extract_feats(self):
        params = {}
        params['output_dir'] = '/Users/bismarck/Downloads/temp_data/scene'
        params['video_path'] = '/Users/bismarck/Downloads/temp_data/videos'
        params['n_frames_steps_2D'] = 10
        model = torchvision.models.resnet101(pretrained=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        extract_feats(params, model, device)

if __name__ == '__main__':
    unittest.main()
