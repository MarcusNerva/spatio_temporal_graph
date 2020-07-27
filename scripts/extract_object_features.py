from collections import OrderedDict
from tqdm import tqdm
import os
import PIL.Image as Image
import glob
import shutil
import subprocess
import torchvision
import torch
import numpy as np
from torch.jit.annotations import Tuple, List, Dict, Optional

from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.faster_rcnn import FasterRCNN, TwoMLPHead, fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
}
EPS = 1e-8


class ObjectExtractor(FasterRCNN):
    """
    ObjectExtractor inherit from FasterRCNN to extract object features of each frames.
    First exploit rpn and roi_head to obtain precise boxes bounding.
    Subsequently, I utilize roi_align to obtain object features.
    """

    def __init__(self, backbone, num_classes=None, box_score_thresh=0.5):
        """
        Args:
            backbone: the network used to compute the features for the model.
            num_classes: number of output classes of the model.
        """
        super(ObjectExtractor, self).__init__(backbone=backbone,
                                              num_classes=num_classes,
                                              box_score_thresh=box_score_thresh)

    def forward(self, images):
        """
        Args:
            images: input images which has been process by transforms.ToTensor()
        Returns:
            box_features_list: list of box_features. len(box_features_list) == number of images.
                                Each element shape of this list is (number of objects, 1024).
            precise_proposals: list of box_boundings. len(precise_proposals) == number of images.
                                Each element shape of this list is (number of objects, 4).
        """
        with torch.no_grad():
            original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
            for img in images:
                val = img.shape[-2:]
                assert len(val) == 2
                original_image_sizes.append((val[0], val[1]))

            images, _ = self.transform(images, None)
            image_shapes = images.image_sizes
            features = self.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([('0', features)])

            proposals, _ = self.rpn(images, features, None)
            detections, _ = self.roi_heads(features, proposals, image_shapes, None)
            precise_proposals = []

            for result in detections:
                precise_proposals.append(result['boxes'])

            box_roi_pool = self.roi_heads.box_roi_pool
            box_head = self.roi_heads.box_head

            box_features = box_roi_pool(features, precise_proposals, image_shapes)
            box_features = box_head(box_features)

            num_boxes_per_image = []
            for pro in precise_proposals:
                num_boxes_per_image.append(pro.shape[0])
            box_features_list = box_features.split(num_boxes_per_image, 0)

            return list(box_features_list), list(precise_proposals)


def iou(box0, box1):
    """
    This function aims to calculate iou value between box0 and box1.
    Args:
        box0: bounding of box0.
        box1: bounding of box1.

    Returns:
        iou value.
    """
    x1, y1, x2, y2 = box0[0], box0[1], box0[2], box0[3]
    x3, y3, x4, y4 = box1[0], box1[1], box1[2], box1[3]

    up, down, lft, rht = max(y1, y3), min(y2, y4), max(x1, x3), min(x2, x4)
    intersection = torch.clamp((down - up), min=0.0) * torch.clamp((rht - lft), min=0.0)
    union = (y2 - y1) * (x2 - x1) + (y4 - y3) * (x4 - x3) - intersection
    return intersection.to(torch.float32) / (union + EPS)


def compute_spatial_matrix(box_boundings):
    """
    This function aims to compute spatial graph of a single frame according to IoU value between objects.

    Args:
        box_boundings: bounding boxes of detected object. boxes.shape == (number of objects, 4)

    Returns:
        G_spatial: spatial graph matrix whose shape is (5, 5).
    """
    obj_nums, _ = box_boundings.shape
    G_spatial = torch.zeros((obj_nums, obj_nums))

    for i in range(obj_nums):
        for j in range(obj_nums):
            G_spatial[i][j] = iou(box_boundings[i], box_boundings[j])

    return G_spatial


def compute_temporal_matrix(box_features_crt, box_features_next):
    """
    This function aims to compute temporal graph between 2 adjacent frames,
    according to cosine similar between object.

    Args:
        box_features_crt: features from current frame. shape == (5, 1024)
        box_features_next: features from next frame. shape == (5, 1024)

    Returns:
         G_temporal: temporal graph matrix whose shape is (5, 5).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    box_features_crt = box_features_crt.to(device)
    box_features_next = box_features_next.to(device)

    crt_norm = torch.norm(box_features_crt, dim=1) + EPS
    next_norm = torch.norm(box_features_next, dim=1) + EPS

    norm_matrix = crt_norm[:, None] * next_norm[None, :]

    cosine_matrix = box_features_crt @ box_features_next.T
    cosine_matrix = cosine_matrix / norm_matrix

    softmax_func = torch.nn.Softmax(dim=1)
    G_temporal = softmax_func(cosine_matrix)

    return G_temporal

def extract_object_features(params, model, device):
    """
    This function aims to extract object features with help of Faster-RCNN,
    which is considered as object detector.
    Faster-RCNN(ResNet50 + FPN backbone, pre-trained on coco 2017) which is exploited here,
    come from torchvision, and is different from Faster-RCNN(ResNeXt101 + FPN backbone, pre-trained on Visual Genome)
    which is mentioned in the paper.

    First, I utilize ffmpeg to extract frames from each video.
    Second, I utilize ObjectExtractor(inherit from Faster-RCNN)
            to obtain detections with high confidence score. For each frame, I take top5 detections.
            I set confidence score threshold for a detection to be considered at 0.5, as mentioned in paper.
    Third, I use iou value between detection boxes to compute G_space matrix, G_space.shape == (5, 5).
           Subsequently, I use cosine similar value to compute pair-wise feature similarity matrix,
           namely, G_time. G_time.shape == (5, 5).
    At last, save these matrix on disk.

    Args:
        params: all the relevant settings.
        model: object extractor.
        device: cpu or gpu.

    Returns:
        None.
    """

    data_dir = params['data_dir']
    dir_fc = params['output_dir']
    video_store = params['video_path']
    n_frames = params['n_frames_per_video']
    max_objects = 5

    assert data_dir is not '', 'please set data_dir!'
    assert dir_fc is not '', 'please set dir_fc!'
    assert video_store is not '', 'please set video_store!'
    assert n_frames != 0, 'please set n_frames_per_video!'

    dir_frame = os.path.join(data_dir, 'frames')
    dir_spatial = os.path.join(dir_fc, 'spatial')
    dir_temporal = os.path.join(dir_fc, 'temporal')

    if not os.path.exists(dir_fc):
        os.mkdir(dir_fc)
    if not os.path.exists(dir_spatial):
        os.mkdir(dir_spatial)
    if not os.path.exists(dir_temporal):
        os.mkdir(dir_temporal)

    print('save video object features to %s' % (dir_fc))
    print('save video spatial features to %s' % (dir_spatial))
    print('save video temporal features to %s' % (dir_temporal))

    video_list = glob.glob(os.path.join(video_store, '*.mp4'))
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    for video in tqdm(video_list):
        video_id = video.split('/')[-1].split('.')[0]
        temp_frame_dir = os.path.join(dir_frame, video_id)

        image_list = sorted(glob.glob(os.path.join(temp_frame_dir, '*.jpg')))
        samples = np.round(np.linspace(0, len(image_list) - 1, n_frames))
        image_list = [image_list[int(sample)] for sample in samples]
        images = []
        for img_path in image_list:
            img = Image.open(img_path)
            img = transform(img)
            images.append(img.to(device))
        feature_list, proposal_list = model(images)
        spatial_list, temporal_list = [], []
        for i in range(n_frames):
            feature, proposal = feature_list[i], proposal_list[i]
            n_objects, feature_dim = feature.shape
            _, proposal_dim = proposal.shape
            if n_objects >= max_objects:
                feature, proposal = feature[:max_objects], proposal[:max_objects]
            else:
                new_feature, new_proposal = torch.zeros(max_objects, feature_dim), torch.zeros(max_objects, proposal_dim)
                new_feature[:n_objects, ...] = feature[:n_objects, ...]
                new_proposal[:n_objects, ...] = proposal[:n_objects, ...]
                feature, proposal = new_feature, new_proposal
            feature_list[i], proposal_list[i] = feature, proposal

        for i in range(n_frames):
            crt_feature, crt_proposal = feature_list[i], proposal_list[i]
            G_spatial = compute_spatial_matrix(crt_proposal)
            spatial_list.append(G_spatial)
            if i < n_frames - 1:
                next_feature, next_proposal = feature_list[i + 1], proposal_list[i + 1]
                G_temporal = compute_temporal_matrix(crt_feature, next_feature)
                temporal_list.append(G_temporal)

        # spatial_features.shape == (10, 5, 5)
        spatial_features = torch.stack(spatial_list, dim=0).cpu().numpy()
        # temporal_features.shape == (9, 5, 5)
        temporal_features = torch.stack(temporal_list, dim=0).cpu().numpy()
        outfile_spatial = os.path.join(dir_spatial, video_id + '.npy')
        outfile_temporal = os.path.join(dir_temporal, video_id + '.npy')
        np.save(outfile_spatial, spatial_features)
        np.save(outfile_temporal, temporal_features)

        shutil.rmtree(temp_frame_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', dest='data_dir', type=str,
                        default='/disks/lilaoshi666/hanhua.ye/spatio_temporal_graph/data',
                        help='directory of storing data')
    parser.add_argument('--output_dir', dest='output_dir', type=str,
                        default='/disks/lilaoshi666/hanhua.ye/spatio_temporal_graph/data/object',
                        help='the directory of storing object features')
    parser.add_argument('--n_frames_per_video', dest='n_frames_per_video', type=int,
                        default=10,
                        help='how many frames should I utilized to extract features')
    parser.add_argument('--video_path', dest='video_path', type=str,
                        default='/disks/lilaoshi666/hanhua.ye/spatio_temporal_graph/data/videos',
                        help='path of dir where holds videos')

    args = parser.parse_args()
    params = vars(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = resnet_fpn_backbone(backbone_name='resnet50', pretrained=True)
    object_extractor = ObjectExtractor(backbone=backbone, num_classes=91, box_score_thresh=0.5)
    state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'], progress=True)
    object_extractor.load_state_dict(state_dict=state_dict)
    object_extractor.eval()
    object_extractor.to(device)

    extract_object_features(params=params, model=object_extractor, device=device)
    print('############extracting object features accomplished!############')