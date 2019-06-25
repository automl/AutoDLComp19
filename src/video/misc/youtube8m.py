import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import csv
import os
import sys
import subprocess
import shutil


TRAIN_LABELS = '/home/dingsda/autodl/krollac-youtube-8m/train_labels.csv'
VALID_LABELS = '/home/dingsda/autodl/krollac-youtube-8m/validate_labels.csv'
TEST_LABES = '/home/dingsda/autodl/krollac-youtube-8m/test_labels.csv'

TRAIN_LINKS = '/home/dingsda/autodl/krollac-youtube-8m/unique_train_links.txt'
VALID_LINKS = '/home/dingsda/autodl/krollac-youtube-8m/unique_val_links.txt'

TRAIN_SLINKS = '/home/dingsda/autodl/krollac-youtube-8m/train_selected_links.csv'
VALID_SLINKS = '/home/dingsda/autodl/krollac-youtube-8m/validate_selected_links.csv'

DOWNLOAD_FOLDER = '/media/dingsda/External/download/frames_youtube8m'

TRAIN_BIAS = 6e8
VALID_BIAS = 1.5e7

NUM_PROCESSES = 8


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


def keep_elem(v_la_elem, abs_num, bias):
    keep_ratio = 0
    for label in v_la_elem[1]:
        keep_ratio = max(keep_ratio, bias/(bias+abs_num[label]**3))

    if np.random.rand() < keep_ratio:
        return True
    else:
        return False


def write_selected_video_links(v_la_n, v_li, slink_path):
    v_la_n.sort()

    with open(slink_path, 'w') as file:
        for v_la_elem in v_la_n:
            video = v_la_elem[0]
            if video in v_li.keys():
                file.write(video + ' ' + v_li[video])


def select_videos(label_path, link_path, slink_path, bias):
    v_la = read_video_labels(label_path)
    v_li = read_video_links(link_path)
    abs_num = abs_number_of_labels(v_la)

    v_la_n = []
    for v_l_elem in v_la:
        if keep_elem(v_l_elem, abs_num, bias):
            v_la_n.append(v_l_elem)
    abs_num_n = abs_number_of_labels(v_la_n)
    plot_abs_number_of_labels(abs_num)
    plot_abs_number_of_labels(abs_num_n)
    write_selected_video_links(v_la_n, v_li, slink_path)

    print('resulting number of videos: ' + str(len(v_la_n)))


##############################################
# upper part: analysis // lower part: download
##############################################


def download_and_convert_parallel(slink_path, download_folder, num_processes):
    p_list = []
    for i in range(num_processes):
        p = mp.Process(target=download_and_convert, args=(slink_path, download_folder, i, num_processes))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()


def download_and_convert(slink_path, download_folder, process_id=0, num_processes=1):
    with open(slink_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE)

        for i, row in enumerate(reader):
            if (i+process_id)%num_processes != 0:
                continue

            # main step: load video in browser, extract download link and store it
            video_id = row[0]
            filename = os.path.join(download_folder, video_id + '.mp4')
            foldername = os.path.join(download_folder, video_id)
            foldername_temp = foldername + '_temp'

            # do not download files twice
            if os.path.isdir(foldername):
                print('File already converted: ' + foldername)
                continue

            if not os.path.exists(foldername_temp):
                os.mkdir(foldername_temp)

            try:
                video_url = row[1]
                print('Found video: ' + video_url)
                download_video(video_url, filename)
                convert_video(filename, foldername, foldername_temp)
            except Exception as err:
                print('-----------------')
                print(err)
                print('-----------------')
            finally:
                cleanup(filename, foldername_temp)


def download_video(video_url, filename):
    if os.path.isfile(filename):
        print('File already exists: ' + filename)
        return

    subprocess.call(['youtube-dl', '-q', '-o', filename, '-f', '133', video_url])
    if os.path.isfile(filename):
        return

    print('second try')
    subprocess.call(['youtube-dl', '-q', '-o', filename, '-f', '134', video_url])
    if os.path.isfile(filename):
        return

    raise Exception('File not downloaded properly: ' + filename)


def convert_video(filename, foldername, foldername_temp):
    dest = os.path.join(foldername_temp, "%04d.jpg")
    command = "ffmpeg -v error -i " + filename + " -y -r 1 " + dest
    os.system(command)
    os.rename(foldername_temp, foldername)


def cleanup(filename, foldername_temp):
    #if os.path.isfile(filename):
    #    os.remove(filename)
    if os.path.exists(foldername_temp):
        shutil.rmtree(foldername_temp)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(arg)
        download_and_convert(TRAIN_SLINKS, DOWNLOAD_FOLDER, int(sys.argv[1]), int(sys.argv[2]))
    else:
        download_and_convert_parallel(TRAIN_SLINKS, DOWNLOAD_FOLDER, NUM_PROCESSES)

    #select_videos(TRAIN_LABELS, TRAIN_LINKS, TRAIN_SLINKS, TRAIN_BIAS)
    #select_videos(VALID_LABELS, VALID_LINKS, VALID_SLINKS, VALID_BIAS)