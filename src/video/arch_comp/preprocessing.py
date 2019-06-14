import cv2
import numpy as np


def get_segment_array(video, segment_length=10, segment_count=10, cut_type="even"):
    video_frames = video.shape[0]
    segment_array = np.zeros([segment_count, segment_length]).astype(int)

    if cut_type == "even":
        # cut the video into evenly spaced short segments
        for i in range(segment_count):
            start_index = np.floor(
                (video_frames - segment_length - 1) * (i / (segment_count - 1))
            )
            end_index = start_index + segment_length
            segment_array[i] = np.arange(start_index, end_index)

    elif cut_type == "random":
        # cut the video into randomly sampled short segments
        start_index_max = np.round(video_frames - segment_length - 1)
        for i in range(segment_count):
            start_index = int(np.random.random() * start_index_max)
            end_index = start_index + segment_length
            segment_array[i] = np.arange(start_index, end_index)
        segment_array.sort(0)

    elif cut_type == "random_within_segment":
        # cut the video into short segments randomly sampled within a certain area
        for i in range(segment_count):
            seg_index_start = np.floor((video_frames - 1) * ((i + 0) / segment_count))
            seg_index_end = (
                np.floor((video_frames - 1) * ((i + 1) / segment_count)) - segment_length
            )
            seg_index_diff = seg_index_end - seg_index_start
            start_index = int(seg_index_start + np.random.random() * seg_index_diff)
            end_index = start_index + segment_length
            segment_array[i] = np.arange(start_index, end_index)

    else:
        raise Exception("unknown cut type: " + str(cut_type))

    return segment_array


def segment_video(video, segment_array):
    return video[segment_array.flatten()]


def resize_video(video, resize_size):
    # desired video size
    video_out = np.empty((video.shape[0], resize_size, resize_size, video.shape[3]))

    for i, image in enumerate(video):
        video_out[i] = cv2.resize(image, (resize_size, resize_size))

    return video_out


def convert_video_to_float(video):
    video_out = video / 255.0
    return video_out
