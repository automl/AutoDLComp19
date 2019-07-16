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

BASE_FOLDER = '/data/aad/video_datasets/youtube8m'


def remove_empty_frame_folders(base_folder, list_file, delete_folder):
    list_file = os.path.join(base_folder, list_file)
    tmp_file = list_file + '.tmp'

    with open(list_file, 'r') as csvfile_input, \
         open(tmp_file, 'w+') as csvfile_output:

        reader = csv.reader(csvfile_input, delimiter=' ')
        writer = csv.writer(csvfile_output, delimiter=' ')

        for i, row in enumerate(reader):
            folder = BASE_FOLDER + row[0]

            num_files = row[1]

            if num_files == '0':
                print(folder)
                if delete_folder:
                    shutil.rmtree(folder)
            else:
                writer.writerow(row)

    os.remove(list_file)
    shutil.move(tmp_file, list_file)


if __name__ == "__main__":
    remove_empty_frame_folders(BASE_FOLDER, 'train_frames.txt', True)
    remove_empty_frame_folders(BASE_FOLDER, 'test_frames.txt', True)



