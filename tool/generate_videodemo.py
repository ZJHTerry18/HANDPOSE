import cv2
from cv2 import VideoWriter
from matplotlib.pyplot import savefig
from tqdm import tqdm
import os
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--id', type=str)
# parser.add_argument('--hand', type=str, choices=['l', 'r'])
# args = parser.parse_args()

def save_video(img_root, output_path, type = range(26)):
    # img_root = 'demo_test2\\'
    FPS = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    size = (960, 480)
    VideoWriter = cv2.VideoWriter(output_path,fourcc, FPS, size)
    imgs = os.listdir(img_root)
    for img in tqdm(imgs):
        # if img.split('_')[0] in type:
        frame = cv2.imread(os.path.join(img_root, img))
        VideoWriter.write(frame)
    VideoWriter.release()

if __name__ == '__main__':
    handtype = 'left' if args.hand == 'l' else 'right'
    imgroot = 'demo_v3_p' + args.id + '_' + handtype
    videoname = 'output_p' + args.id + '_' + handtype + '.mp4'
    save_video(img_root=imgroot, output_path='D:\Workspace\LeapMotion\leapHandpose\leapHandpose\dataset_v3\\video_demos\\' + videoname)