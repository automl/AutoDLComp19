import csv
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import cv2
import glob

import matplotlib.pyplot as plt
import numpy as np

BASE_FOLDER = '/media/dingsda/External/datasets/youtube8m/'

TRAIN_LABELS = os.path.join(BASE_FOLDER, 'train_labels.csv')
VALID_LABELS = os.path.join(BASE_FOLDER, 'validate_labels.csv')

TRAIN_LINKS = os.path.join(BASE_FOLDER, 'unique_train_links.txt')
VALID_LINKS = os.path.join(BASE_FOLDER, 'unique_val_links.txt')

TRAIN_SLINKS = os.path.join(BASE_FOLDER, 'train_selected_links.csv')
VALID_SLINKS = os.path.join(BASE_FOLDER, 'validate_selected_links.csv')

TRAIN_FILE = os.path.join(BASE_FOLDER, 'train.txt')
VALID_FILE = os.path.join(BASE_FOLDER, 'test.txt')

DOWNLOAD_FOLDER = os.path.join(BASE_FOLDER, 'download')
VIDEO_FOLDER = os.path.join(BASE_FOLDER, 'videos')
DALI_FOLDER = os.path.join(BASE_FOLDER, 'dali')
FAILED_FOLDER = os.path.join(BASE_FOLDER, 'failed')
TEMP_FOLDER = os.path.join(BASE_FOLDER, 'temp')
FRAME_FOLDER = os.path.join(BASE_FOLDER, 'frames')


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



def convert_to_frames_parallel(output_path, video_folder, frame_folder, process_start, process_end, num_processes):
    p_list = []
    for i in range(process_start, process_end):
        p = mp.Process(
            target=convert_to_frames,
            args=(output_path, video_folder, frame_folder, i, num_processes
            )
        )
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()



def convert_to_frames(output_path, video_folder, frame_folder, process_id=0, num_processes=1):
    '''
    convert videos to frame level
    '''
    # recursively parse root_dir

    print('process id ' + str(process_id))
    print('num processes ' + str(num_processes))

    with open(output_path, 'r') as csvfile_output:

        reader = csv.reader(csvfile_output, delimiter=' ', quoting=csv.QUOTE_NONE)

        for i, d_row in enumerate(reader):
            video_id = d_row[0].split('/')[-1]

            file_path = os.path.join(video_folder, video_id)

            file_name = os.path.basename(file_path)
            file_base = os.path.splitext(file_name)[0]
            file_ext = os.path.splitext(file_name)[1]

            dest_folder = os.path.join(frame_folder, file_base)
            dest_folder_temp = os.path.join(frame_folder, file_base+'_temp')

            # use non-conflicting indices for every process
            if (i+process_id)%num_processes != 0:
                continue

            if os.path.exists(dest_folder):
                continue

            # only convert files with specific ending
            if file_ext == ".mp4":
                try:
                    print(dest_folder)

                    if not os.path.exists(dest_folder_temp):
                        os.mkdir(dest_folder_temp)

                    dest = os.path.join(dest_folder_temp, "%04d.jpg")
                    command = "ffmpeg -v error -i " + file_path + " -y " + dest
                    os.system(command)
                    os.rename(dest_folder_temp, dest_folder)
                except Exception as err:
                    print('-----------------')
                    print(err)
                    if os.path.exists(dest_folder_temp):
                        shutil.rmtree(dest_folder_temp)
                    print('-----------------')


def create_metadata_parallel(label_path, video_folder, output_path, process_start, process_end, num_processes):
    p_list = []
    for i in range(process_start, process_end):
        p = mp.Process(
            target=create_metadata,
            args=(
                label_path, video_folder, output_path, i, num_processes
            )
        )
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()


def create_metadata(label_path, video_folder, output_path, process_id=0, num_processes=1):
    output_path = output_path + '_' + str(process_id)

    with open(label_path, 'r') as csvfile_dataset, \
         open(output_path, 'w+') as csvfile_output:

        d_reader = csv.reader(csvfile_dataset, delimiter=',', quoting=csv.QUOTE_NONE)
        o_writer = csv.writer(csvfile_output, delimiter=' ')

        for i, d_row in enumerate(d_reader):
            # write a short progress message
            if i % 1e5 == 0:
                print(i)

            if (i + process_id) % num_processes != 0:
                continue

            video_id = d_row[0]
            file_path = os.path.join(video_folder, str(video_id))
            file_path = os.path.join(file_path, str(video_id) + '.mp4')

            if not os.path.isfile(file_path):
                continue

            try:
                cap = cv2.VideoCapture(file_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                print(file_path)

                meta_list = []
                for label in d_row[1].split(' '):
                    meta_list.append(label)
                    meta_list.append(1)

                o_writer.writerow([video_id, str(frame_count)] + meta_list)
            except Exception as err:
                print('-----------------')
                print(err)
                print('-----------------')


def merge_metadata(output_path):
    split_files = glob.glob(output_path + '_*')

    with open(output_path,'w+') as wf:
        for f in split_files:
            with open(f,'r') as fd:
                shutil.copyfileobj(fd, wf)


def dali_parallel(video_folder, dali_folder, label_path, process_start, process_end, num_processes):
    p_list = []
    for i in range(process_start, process_end):
        p = mp.Process(
            target=dali,
            args=(video_folder, dali_folder, label_path, i, num_processes
            )
        )
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()


def dali(video_folder, dali_folder, label_path, process_id=0, num_processes=1):
    if not os.path.isdir(dali_folder):
        os.mkdir(dali_folder)

    with open(label_path, 'r') as csvfile_dataset:
        d_reader = csv.reader(csvfile_dataset, delimiter=',', quoting=csv.QUOTE_NONE)

        for i, d_row in enumerate(d_reader):
            # write a short progress message
            if i % 1e5 == 0:
                print(i)

            if (i + process_id) % num_processes != 0:
                continue

            video_id = d_row[0]
            file_path = os.path.join(video_folder, str(video_id)+'.mp4')

            if not os.path.isfile(file_path):
                continue

            print(file_path)

            video_folder_new = os.path.join(dali_folder, video_id)
            file_path_new = os.path.join(video_folder_new, str(video_id)+'.mp4')

            if not os.path.isdir(video_folder_new):
                os.mkdir(video_folder_new)

            shutil.move(file_path, file_path_new)


def dali_replace_paths(output_path):
    output_path_new = output_path + '_new'
    with open(output_path, 'r') as csvfile, \
        open(output_path_new, 'w+') as csvfile_new:
            reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE)
            writer = csv.writer(csvfile_new, delimiter=' ')

            for i, row in enumerate(reader):
                if i % 1e5 == 0:
                    print(i)

                row[0] = row[0].split('/')[2][:-4]
                writer.writerow(row)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(arg)
        #dali_parallel(VIDEO_FOLDER, DALI_FOLDER, TRAIN_LABELS, int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        #convert_to_frames_parallel(TRAIN_FILE, VIDEO_FOLDER, FRAME_FOLDER, int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        pass
        #download_and_convert_parallel(TRAIN_SLINKS, DOWNLOAD_FOLDER, VIDEO_FOLDER, FAILED_FOLDER, TEMP_FOLDER, int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

    dali_replace_paths(VALID_FILE)
    # merge_metadata(TRAIN_FILE)

    # select_videos(TRAIN_LABELS, TRAIN_LINKS, TRAIN_SLINKS)
    # select_videos(VALID_LABELS, VALID_LINKS, VALID_SLINKS)

    #convert_to_frames(TRAIN_FILE, VIDEO_FOLDER, FRAME_FOLDER)

