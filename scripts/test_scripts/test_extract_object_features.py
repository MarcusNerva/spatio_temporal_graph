import sys
sys.path.append('/Users/bismarck/PycharmProjects/spatio_temporal_graph/scripts/')
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')
print(sys.path)


from scripts import extract_frames
from scripts.extract_object_features import extract_frames
from scripts.extract_object_features import iou
from scripts.extract_object_features import compute_spatial_matrix
from scripts.extract_object_features import compute_temporal_matrix
from scripts.extract_object_features import extract_object_features

import unittest
import torch
import numpy as np

EPS = 1e-8
def sgn(x):
    if abs(x) < 1e-8: return 0
    elif x > 0: return 1
    return -1

class TestExtractObjectFeatures(unittest.TestCase):

    # def test_extract_frames(self):
    #     video_path = '/Users/bismarck/Downloads/temp_data/videos/video0.mp4'
    #     dst_path = '/Users/bismarck/Downloads/temp_data/object/frames'
    #     extract_frames(video_path, dst_path)

    def test_iou01(self):

        box0, box1 = torch.tensor([1, 1, 3, 3]), torch.tensor([2, 0, 4, 2])
        self.assertEqual(sgn(iou(box0, box1).item() - 1.0 / (7.0 + EPS)), 0)

    def test_iou02(self):
        box2, box3 = torch.tensor([1, 1, 3, 3]), torch.tensor([0, 0, 4, 2])
        self.assertEqual(sgn(iou(box2, box3).item() - 2. / (10. + EPS)), 0)

    def test_iou03(self):
        box4, box5 = torch.tensor([0, 0, 4, 4]), torch.tensor([1, 1, 3, 3])
        self.assertEqual(sgn(iou(box4, box5).item() - 4. / (16. + EPS)), 0)

    def test_iou04(self):
        box6, box7 = torch.tensor([2, 0, 4, 2]), torch.tensor([0, 2, 2, 4])
        self.assertEqual(sgn(iou(box6, box7).item() - 0. / (8. + EPS)), 0)

    def test_iou05(self):
        box8, box9 = torch.tensor([  36.9815,   51.5072,  692.9578,  788.9585]), torch.tensor([ 763.2416,  652.5604,  894.3409,  793.9498])
        print(iou(box8, box9).item())

    def test_compute_spatial_matrix(self):
        # boxes = torch.stack([torch.tensor([1, 1, 3, 3]),
        #                    torch.tensor([2, 0, 4, 2]),
        #                    torch.tensor([0, 0, 4, 2]),
        #                    torch.tensor([0, 0, 4, 4]),
        #                    torch.tensor([0, 2, 2, 4])], dim=0)
        boxes = torch.tensor([[  36.9815,   51.5072,  692.9578,  788.9585],
        [ 459.8420,  433.7843,  964.5193,  796.9225],
        [ 389.9184,    6.7380, 1020.9323,  759.6011],
        [ 763.2416,  652.5604,  894.3409,  793.9498],
        [ 736.4373,  544.3269,  969.4111,  787.8588]])
        G_spatial = compute_spatial_matrix(box_boundings=boxes)
        # answer = np.stack([np.array([4./(4.+EPS), 1./(7.+EPS), 2./(10.+EPS), 4./(16.+EPS), 1./(7.+EPS)]),
        #                    np.array([1./(7.+EPS), 4./(4.+EPS), 4./(8.+EPS), 4./(16.+EPS), 0.]),
        #                    np.array([2./(10.+EPS), 4./(8.+EPS), 8./(8.+EPS), 8./(16.+EPS), 0.]),
        #                    np.array([4./(16.+EPS), 4./(16.+EPS), 8./(16.+EPS), 16./(16.+EPS), 4./(16.+EPS)]),
        #                    np.array([1./(7.+EPS), 0., 0., 4./(16.+EPS), 4./(4.+EPS)])], axis=1)
        for i in range(5):
            for j in range(5):
                # if sgn(G_spatial[i][j].item() - answer[i][j]) != 0:
                #     print("WARNING!!!!!")
                print(G_spatial[i][j].item())

    def test_compute_temporal_matrix(self):
        box_features_crt = torch.ones((5, 1024))
        box_features_next = torch.ones((5, 1024))
        G_temporal = compute_temporal_matrix(box_features_crt, box_features_next)
        answer = np.ones((5, 5))
        for i in range(5):
            for j in range(5):
                if sgn(G_temporal[i][j].item() - answer[i][j]) != 0:
                    print(G_temporal[i][j].item())

    # def test_extract_object_features(self):
    #     pass


if __name__ == '__main__':
    unittest.main()