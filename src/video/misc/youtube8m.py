import csv
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

BASE_FOLDER = '/media/dingsda/External/datasets/youtube8m/'

TRAIN_LABELS = os.path.join(BASE_FOLDER, 'train_labels.csv')
VALID_LABELS = os.path.join(BASE_FOLDER, 'validate_labels.csv')
TEST_LABES = os.path.join(BASE_FOLDER, 'test_labels.csv')

TRAIN_LINKS = os.path.join(BASE_FOLDER, 'unique_train_links.txt')
VALID_LINKS = os.path.join(BASE_FOLDER, 'unique_val_links.txt')

TRAIN_SLINKS = os.path.join(BASE_FOLDER, 'train_selected_links.csv')
VALID_SLINKS = os.path.join(BASE_FOLDER, 'validate_selected_links.csv')

DOWNLOAD_FOLDER = os.path.join(BASE_FOLDER, 'download')
VIDEO_FOLDER = os.path.join(BASE_FOLDER, 'videos')
FAILED_FOLDER = os.path.join(BASE_FOLDER, 'failed')
TEMP_FOLDER = os.path.join(BASE_FOLDER, 'temp')

# percentage of how many files to store in the subset
PERC = 1


def read_video_labels(label_path):
    v_la = []
    with open(label_path, 'r') as file:
        f = file.readlines()
        for line in f:
            video = line.split(',')[0]
            labels = list(map(int, line.split(',')[1].rstrip().split(' ')))
            v_la.append((video, labels))

    return v_la


def read_video_links(link_path):
    v_li = {}
    with open(link_path, 'r') as file:
        f = file.readlines()
        for line in f:
            video = line.split(' ')[0]
            link = line.split(' ')[1]
            v_li[video] = link
    return v_li


def avg_number_of_labels_per_video(v_l):
    avg_num = np.zeros([len(v_l)])

    for i in range(len(v_l)):
        avg_num[i] = len(v_l[i][1])

    return sum(avg_num) / len(v_l)


def abs_number_of_labels(v_la):
    abs_num = {}

    for elem in v_la:
        for label in elem[1]:
            if label not in abs_num.keys():
                abs_num[label] = 1
            else:
                abs_num[label] += 1

    return abs_num


def plot_abs_number_of_labels(abs_num):
    lst = [v for k, v in abs_num.items()]
    lst.sort()
    plt.plot(lst)
    plt.grid(True)
    plt.yscale('log')
    plt.xlabel("label index")
    plt.ylabel("label occurrence")
    plt.show()


def keep_elem():
    # if np.random.rand() < PERC:
    #     return True
    #
    # return False
    return True


def write_selected_video_links(v_la_n, v_li, slink_path):
    #v_la_n.sort()
    random.shuffle(v_la_n)

    with open(slink_path, 'w') as file:
        for v_la_elem in v_la_n:
            video = v_la_elem[0]
            if video in v_li.keys():
                file.write(video + ' ' + v_li[video])


def select_videos(label_path, link_path, slink_path):
    v_la = read_video_labels(label_path)
    v_li = read_video_links(link_path)
    abs_num = abs_number_of_labels(v_la)

    v_la_n = []
    for v_l_elem in v_la:
        if keep_elem():
            v_la_n.append(v_l_elem)
    abs_num_n = abs_number_of_labels(v_la_n)
    plot_abs_number_of_labels(abs_num)
    plot_abs_number_of_labels(abs_num_n)
    write_selected_video_links(v_la_n, v_li, slink_path)

    print('resulting number of videos: ' + str(len(v_la_n)))


##############################################
# upper part: analysis // lower part: download
##############################################


def download_and_convert_parallel(
    slink_path, download_folder, video_folder, failed_folder, temp_folder,
    process_start, process_end, num_processes
):
    p_list = []
    for i in range(process_start, process_end):
        p = mp.Process(
            target=download_and_convert,
            args=(
                slink_path, download_folder, video_folder, failed_folder, temp_folder, i,
                num_processes
            )
        )
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()


def download_and_convert(
    slink_path,
    download_folder,
    video_folder,
    failed_folder,
    temp_folder,
    process_id=0,
    num_processes=1
):
    for folder in [download_folder, video_folder, failed_folder, temp_folder]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    with open(slink_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE)

        for i, row in enumerate(reader):
            if (i + process_id) % num_processes != 0:
                continue

            video_id = row[0]
            download_name = os.path.join(download_folder, video_id + '.mp4')
            video_name = os.path.join(video_folder, video_id + '.mp4')
            failed_name = os.path.join(failed_folder, video_id + '.mp4')
            temp_name = os.path.join(temp_folder, video_id + '.mp4')

            if os.path.isfile(video_name) or os.path.isfile(failed_name):
                print('ID already treated: ' + str(video_id))
                continue

            try:
                video_url = row[1]
                #print('Found video: ' + video_url)
                print('download')
                download_video(video_url, download_name)
                print('convert')
                convert_video(download_name, video_name, temp_name)
            except Exception as err:
                print('-----------------')
                print(err)
                if not os.path.exists(failed_name):
                    os.mknod(failed_name)
                print('-----------------')


def download_video(video_url, download_name):
    subprocess.call(['youtube-dl', '-q', '-o', download_name, '-f', '133', video_url])
    if os.path.isfile(download_name):
        return

    print('second try')
    subprocess.call(['youtube-dl', '-q', '-o', download_name, '-f', '134', video_url])
    if os.path.isfile(download_name):
        return

    raise Exception('File not downloaded properly: ' + download_name)


def convert_video(download_name, video_name, temp_name):
    command = "ffmpeg -v error -i " + download_name + " -an -vf select='not(mod(n\,24)),setpts=N/FRAME_RATE/TB' -r 24 " + temp_name
    os.system(command)
    os.rename(temp_name, video_name)
    os.remove(download_name)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(arg)
        download_and_convert_parallel(TRAIN_SLINKS, DOWNLOAD_FOLDER, VIDEO_FOLDER, FAILED_FOLDER, TEMP_FOLDER, int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

    # select_videos(TRAIN_LABELS, TRAIN_LINKS, TRAIN_SLINKS)
    # select_videos(VALID_LABELS, VALID_LINKS, VALID_SLINKS)

