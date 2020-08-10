import os
import shutil
import subprocess
import glob
from tqdm import tqdm

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', dest='output_dir', type=str,
                        default='/disks/lilaoshi666/hanhua.ye/spatio_temporal_graph/data/frames',
                        help='directory to store features')
    parser.add_argument('--video_path', dest='video_path', type=str,
                        default='/disks/lilaoshi666/hanhua.ye/spatio_temporal_graph/data/videos',
                        help='path to video dataset')

    args = parser.parse_args()
    params = vars(args)
    video_list = glob.glob(os.path.join(params['video_path'], '*.mp4'))

    for video in tqdm(video_list):
        video_id = video.split('/')[-1].split('.')[0]
        dst = os.path.join(params['output_dir'], video_id)
        extract_frames(video, dst)

    print('##############__Frames Extraction Completed__##############')