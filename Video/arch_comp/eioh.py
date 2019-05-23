import os
import glob
import cv2
import numpy as np
from subprocess import call


def load_video_file(file='./video.mp4'):
    '''
    Load a video from a video file path

    :param file: path to video file
    :return: video as numpy array
    '''

    cap = cv2.VideoCapture(file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    return buf


def load_image_dir(root_dir='.'):
    '''
    Load a video from a directory containing images

    :param root_dir: directory containing the images
    :return: video as numpy array
    '''

    imgs = []

    for file in sorted(os.listdir(root_dir)):
        abs_file = os.path.join(os.path.abspath(root_dir), file)

        if os.path.isfile(abs_file):
            imgs.append(cv2.imread(abs_file, cv2.IMREAD_COLOR))
        else:
            raise Exception('directory should only contain files: ' + root_dir)

    return np.asarray(imgs)


def convert_videos_to_frames(root_dir, delete_videos=True):
    '''
    Convert all videos in a given directory and its subdirectories to a list of images
    If delete_videos == True, the original videos will be deleted after conversion

    :param root_dir: base directory
    :param delete_videos: delete original videos
    '''

    # recursively parse root_dir
    for file_path in glob.iglob(root_dir + '**/**', recursive=True):
        file_name = os.path.basename(file_path)
        file_stem = os.path.splitext(file_path)[0]
        file_ext = os.path.splitext(file_name)[1]

        # only convert files with specific ending
        if file_ext == '.avi' or file_ext == '.mp4':
            if not os.path.exists(file_stem):
                os.mkdir(file_stem)
            dest = os.path.join(file_stem, '%04d.jpg')
            call(["ffmpeg", "-y", "-i", file_path, dest])
            if delete_videos:
                os.remove(file_path)


if __name__ == "__main__":
    convert_videos_to_frames('/home/dingsda/autodl/data/ucf101_frames')
