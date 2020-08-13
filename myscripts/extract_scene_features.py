import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os
import argparse

import torch
import torchvision
import cv2
C, H, W = 3, 224, 224

def process_frames(frames_path):
    """
    This function aims to resize frames and center crop these frames to (224, 224, 3).

    Args:
        frames_path: the path of frames.

    Returns: None

    """
    global C, H, W

    frames_path_list = glob.glob(os.path.join(frames_path, '*.jpg'))
    for frame_path in frames_path_list:
        frame = cv2.imread(frame_path)
        h, w = frame.shape[:2]

        short = min(h, w)
        scale = 224. / short
        frame = cv2.resize(frame, dsize=(0, 0), fx=scale, fy=scale)

        crt_h, crt_w = frame.shape[:2]
        h_top = 0 if crt_h <= H else (crt_h - H) // 2
        w_lft = 0 if crt_w <= W else (crt_w - W) // 2
        frame = frame[h_top: h_top + H, w_lft: w_lft + W]
        cv2.imwrite(frame_path, frame)


def extract_frames(video, dst):
    """
    This function aims to extract frames from videos.

    Args:
        video: path of input video
        dst: the directory for saving extracted frames

    Returns: None

    """
    with open(os.devnull, 'w') as ffmpeg_log:
        if os.path.exists(dst):
            print('clean up ' + dst)
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ['ffmpeg', '-y', '-i', video, '-r', '25', '-vf', 'scale=400:300', '-qscale:v', '2',
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command, stdout=ffmpeg_log, stderr=ffmpeg_log)


def compute_optical_flow(prev_frame, crt_frame):
    """
    This function aims to compute optical flow features of videos by DualTVL1 algorithm.

    Args:
        prev_frame: the previous frame
        crt_frame: current frame

    Returns:
        x_flow: the optical flow of x axis
        y_flow: the optical flow of y axis
    """
    # adapt dense optical flow here
    dualTVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    optical_flow = dualTVL1.calc(prev_frame, crt_frame, None)

    x_flow = optical_flow[..., 0]
    y_flow = optical_flow[..., 1]

    return x_flow, y_flow

def save_flows(x_flow, y_flow, flow_path):
    """
    This function aims to save optical flow on disk.
    Every optical flow frame is scaled to [0, 255].

    Args:
        x_flow: optical flow frame on x axis.
        y_flow: optical flow frame on y axis.
        flow_path: the path for saving.

    Returns: None

    """
    # process x_flow
    up_bound, low_bound = np.max(x_flow), np.min(x_flow)
    total_range = up_bound - low_bound
    x_flow -= low_bound
    x_flow /= total_range
    x_flow *= 255
    x_flow[x_flow > 255] = 255
    x_flow[x_flow < 0] = 0
    x_flow = x_flow.astype(np.uint8)

    # process y_flow
    up_bound, low_bound = np.max(y_flow), np.min(y_flow)
    total_range = up_bound - low_bound
    y_flow -= low_bound
    y_flow /= total_range
    y_flow *= 255
    y_flow[y_flow > 255] = 255
    y_flow[y_flow < 0] = 0
    y_flow = y_flow.astype(np.uint8)

    cv2.imwrite(flow_path + '_x.jpg', x_flow)
    cv2.imwrite(flow_path + '_y.jpg', y_flow)

def extract_feats(params, model, device):
    """
    This function aims to extract 2D features by utilizing ResNet101
    (As paper request, I uniformly sampled 10 frames from each video),
    and extract optical flow features. And save these features on disk.

    Args:
        params: all kinds of relevant settings.
        model: model for extracting 2D features. ResNet101 here.
        load_image_fn: function to load images.
        device: run on cpu or gpu.

    Returns:

    """
    global C, H, W
    model.eval()

    dir_fc = params['output_dir']
    data_dir = params['data_dir']
    dir_optical_fc = os.path.join(dir_fc, 'optical_flow')
    dir_2d_fc = os.path.join(dir_fc, '2d')
    dir_frame_fc = os.path.join(data_dir, 'frames')

    if not os.path.exists(dir_fc):
        os.mkdir(dir_fc)
    if not os.path.exists(dir_optical_fc):
        os.mkdir(dir_optical_fc)
    if not os.path.exists(dir_2d_fc):
        os.mkdir(dir_2d_fc)

    print('save video feats to %s' % (dir_fc))
    print('save 2d feats to %s' % (dir_2d_fc))
    print('save optical flow frames to %s' % (dir_optical_fc))
    video_list = glob.glob(os.path.join(params['video_path'], '*.mp4'))

    for video in tqdm(video_list):
        video_id = video.split('/')[-1].split('.')[0]
        dst = os.path.join(dir_frame_fc, video_id)
        extract_frames(video, dst)
        process_frames(dst)

        # extract ResNet101 2D features
        image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
        image_list_cpy = image_list.copy()
        samples = np.round(np.linspace(0, len(image_list) - 1, params['n_frame_steps_2D']))
        image_list = [image_list[int(sample)] for sample in samples]
        images = torch.zeros(len(image_list), C, H, W)
        for iImg in range(len(image_list)):
            img = cv2.imread(image_list[iImg])
            img = img.transpose([2, 0, 1])
            images[iImg] = torch.from_numpy(img)
        with torch.no_grad():
            fc_feats = model(images.to(device)).squeeze()
        img_feats = fc_feats.cpu().numpy()
        outfile = os.path.join(dir_2d_fc, video_id + '.npy')
        np.save(outfile, img_feats)

        """
        # extract optical flow features
        example_frame = cv2.imread(image_list_cpy[0])
        H, W = example_frame.shape[:2]
        prev = np.zeros((H, W)).astype(np.uint8)
        for i in range(len(image_list_cpy)):
            crt = cv2.imread(image_list_cpy[i])
            crt = cv2.cvtColor(crt, cv2.COLOR_BGR2GRAY)
            x_flow, y_flow = compute_optical_flow(prev_frame=prev, crt_frame=crt)
            flow_path = os.path.join(dir_optical_fc, video_id + '_' + str(i))
            save_flows(x_flow=x_flow, y_flow=y_flow, flow_path=flow_path)
            prev = crt
        """
        shutil.rmtree(dst)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', dest='data_dir', type=str,
                        default='/disks/lilaoshi666/hanhua.ye/spatio_temporal_graph/data',
                        help='directory of storing data')
    parser.add_argument('--output_dir', dest='output_dir', type=str,
                        default='/disks/lilaoshi666/hanhua.ye/spatio_temporal_graph/data/scene',
                        help='directory to store features')
    parser.add_argument('--n_frame_steps_2D', dest='n_frame_steps_2D', type=int,
                        default=10,
                        help='how many frames to sample per video')
    parser.add_argument('--video_path', dest='video_path', type=str,
                        default='/disks/lilaoshi666/hanhua.ye/spatio_temporal_graph/data/videos',
                        help='path to video dataset')

    args = parser.parse_args()
    params = vars(args)

    resnet101 = torchvision.models.resnet101(pretrained=True)
    resnet101 = resnet101.to(device)
    extract_feats(params=params, model=resnet101, device=device)

    print('############extracting scene features accomplished!############')
