import cv2
import numpy as np
from ofdis import pyx_flow, visualize_flo

import time
import matplotlib.pylab as plt
from model_pool import CombiNet

import torchvision.models as models
import datasets as datasets
import torch.optim as optim
import torch.nn as nn

import torch
import torchvision


# pip install opencv-python
# pip install opencv-contrib-python

def read_video(filename):
    cap = cv2.VideoCapture(filename)
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


def get_segment_array(video, segment_length=10, segment_count=10, cut_type='even'):
    video_frames = video.shape[0]
    segment_array = np.zeros([segment_count, segment_length]).astype(int)

    if segment_length*segment_count >= video_frames:
        # if video length is shorter than combined segment length, return whole video (-1 because of OF calculation)
        return np.arange(video_frames)

    elif cut_type == 'even':
        # cut the video into evenly spaced short segments
        for i in range(segment_count):
            start_index = np.floor((video_frames-segment_length-1)*(i/(segment_count-1)))
            end_index = start_index + segment_length
            segment_array[i] = np.arange(start_index, end_index)

    elif cut_type == 'random':
        # cut the video into randomly sampled short segments
        start_index_max = np.round(video_frames-segment_length-1)
        for i in range(segment_count):
            start_index = int(np.random.random()*start_index_max)
            end_index = start_index + segment_length
            segment_array[i] = np.arange(start_index, end_index)
        segment_array.sort(0)

    elif cut_type == 'random_within_segment':
        # cut the video into short segments randomly sampled within a certain area
        for i in range(segment_count):
            seg_index_start = np.round((video_frames-1)*((i+0)/segment_count))
            seg_index_end   = np.round((video_frames-1)*((i+1)/segment_count)) - segment_length
            seg_index_diff  = seg_index_end-seg_index_start
            start_index = int(seg_index_start + np.random.random() * seg_index_diff)
            end_index = start_index + segment_length
            segment_array[i] = np.arange(start_index, end_index)

    else:
        raise Exception('unknown cut type: ' +  str(cut_type))

    return segment_array


def segment_element(elem, segment_array):
    return elem[segment_array.flatten()]


def resize_video(video, resize_size):
    # desired video size
    video_out = np.empty((video.shape[0], resize_size, resize_size, video.shape[3]), np.dtype('uint8'))

    for i, image in enumerate(video):
        video_out[i] = cv2.resize(image,(resize_size,resize_size))

    return video_out


def calc_optical_flow(video, segment_array):
    video_out = np.empty((video.shape[0], video.shape[1], video.shape[2], 2), np.dtype('uint8'))

    i=0
    for segment in segment_array:
        seg_len = len(segment)

        if seg_len <= 2:
            raise Exception('segment length too small, cannot calculate optical flow')

        for k in range(seg_len):
            if k == seg_len-1:  # last segment frame
                # TODO: fix dirty hack, last OF image of every segment is similar to secondlast one
                video_out[i] = video_out[i-1]
            else:
                video_out[i] = pyx_flow.optical_fn(video[i],video[i+1])
            i+=1

    return video_out


def calc_features(video, type):
    if type == 0:
        desc = cv2.xfeatures2d.SURF_create()
    elif type == 1:
        desc = cv2.ORB_create()
    elif type == 2:
        desc = cv2.BRISK_create()
    elif type == 3:
        desc = cv2.AKAZE_create()

    dcs_list = []
    dcs_array = np.empty([0,32])
    for image in video:
        kps, dcs = desc.detectAndCompute(image, None)
        dcs_list.append(dcs)
        np.append(dcs_array,dcs,0)

    return dcs_list, dcs_array


def calc_hog(video):
    hog = cv2.HOGDescriptor()

    for image in video:
        h = hog.compute(image)


def calc_mser(video):
    mser = cv2.MSER_create()

    t1 = time.time()
    for image in video:
        regions = mser.detectRegions(image)


def reshape_data_for_model(x):
    # convert to tensor
    y = torch.Tensor(x)
    # realign axis
    y = y.permute(0, 3, 1, 2)
    # add channels if channel dimension does not match
    if y.shape[1] < 3:
        y = torch.cat((y, torch.zeros([y.shape[0], 3 - y.shape[1], y.shape[2], y.shape[3]])), 1)

    return y



