import csv
import os
import multiprocessing as mp
import subprocess
import glob
import sys
import shutil
from itertools import chain


# path to the original autotag file
AUTOTAGS = '/media/dingsda/External/datasets/yfcc100m/unzip/yfcc100m_autotags'
# path to the original dataset file
DATASET = '/media/dingsda/External/datasets/yfcc100m/unzip/yfcc100m_dataset'

# path to the file containing all possible labels as a list
LABEL_LIST = '/media/dingsda/External/datasets/yfcc100m/log/yfcc100m_label'
# overall resulting file with images+labels
METADATA = '/media/dingsda/External/datasets/yfcc100m/log/yfcc100m_metadata'
# path to the video download folder
IMAGE_FOLDER = '/media/dingsda/External/datasets/yfcc100m/images'
# path to the video download folder
TEMP_FOLDER = '/media/dingsda/External/datasets/yfcc100m/temp'
# path to the video download folder
FAILED_FOLDER = '/media/dingsda/External/datasets/yfcc100m/failed'

# guess what
NUM_PROCESSES = 8

csv.field_size_limit(100000000)




def download_parallel(dataset_path, image_folder, temp_folder, failed_folder, num_processes):
    '''
    does the same as extract_download_links(), but in parallel
    '''
    p_list = []
    for i in range(num_processes):
        p = mp.Process(target=download, args=(dataset_path, image_folder, temp_folder, failed_folder, i, num_processes))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()



def download(dataset_path, image_folder, temp_folder, failed_folder, process_id=0, num_processes=1):
    '''
    extract the download links from the yfcc100m dataset and store them in a file
    '''
    for folder in [image_folder, failed_folder, temp_folder]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    with open(dataset_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)

        # find all labels and store them in label_dict
        for i, row in enumerate(reader):
            # write a short progress message
            if i % 1e6 == 0:
                print(i)

            # use non-conflicting indices for every process
            if (i+process_id)%num_processes != 0:
                continue

            # if download path ends with '.jpg', ignore it
            if os.path.splitext(row[16])[1] == '.jpg':
                id = row[1]
                temp_name =  os.path.join(temp_folder, id + '.jpg')
                image_name = os.path.join(image_folder, id + '.jpg')
                failed_name = os.path.join(failed_folder, id + '.jpg')

                # do not download files twice
                if os.path.isfile(image_name) or os.path.isfile(failed_name):
                    continue

                try:
                    image_url = row[16]
                    ret_val = subprocess.call(['wget', '-nv', '-t1', image_url, '-O', temp_name])
                    if ret_val != 0:
                        raise Exception('download not finished properly: ' + image_url)
                    os.rename(temp_name, image_name)
                except Exception as err:
                    print(err)
                    if not os.path.exists(failed_name):
                        os.mknod(failed_name)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(arg)
        download(DATASET, IMAGE_FOLDER, TEMP_FOLDER, FAILED_FOLDER, int(sys.argv[1]), int(sys.argv[2]))
    else:
        download_parallel(DATASET, IMAGE_FOLDER, TEMP_FOLDER, FAILED_FOLDER, NUM_PROCESSES)

