import csv
import os
import multiprocessing as mp
import subprocess
import glob
import sys
import shutil
from itertools import chain


BASE_FOLDER = '/media/dingsda/External/datasets/yfcc100m/'

# path to the original autotag file
AUTOTAGS = os.path.join(BASE_FOLDER,'unzip/yfcc100m_autotags')
# path to the original dataset file
DATASET = os.path.join(BASE_FOLDER,'unzip/yfcc100m_dataset')

# path to the file containing all possible labels as a list
LABEL_LIST = os.path.join(BASE_FOLDER,'yfcc100m_label')
# overall resulting file with images+labels
METADATA = os.path.join(BASE_FOLDER,'yfcc100m_images_metadata')
# path to the video download folder
IMAGE_FOLDER = os.path.join(BASE_FOLDER,'images/')
# path to the video download folder
TEMP_FOLDER = os.path.join(BASE_FOLDER,'temp/')
# path to the video download folder
FAILED_FOLDER = os.path.join(BASE_FOLDER,'failed/')

# guess what
NUM_PROCESSES = 8

csv.field_size_limit(100000000)

# test ratio (1/N)
TEST_RATIO = 5


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


def create_metadata_parallel(autotags_path, dataset_path, image_folder, label_list_path, metadata_path, process_start, process_end, num_processes):
    p_list = []
    for i in range(process_start, process_end):
        p = mp.Process(
            target=create_metadata,
            args=(autotags_path, dataset_path, image_folder, label_list_path, metadata_path, i, num_processes
            )
        )
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()



def create_metadata(autotags_path, dataset_path, image_folder, label_list_path, metadata_path, process_id=0, num_processes=1):
    '''
    create metadata based on downloaded files
    '''
    label_dict = {}
    metadata_path = metadata_path + '_' + str(process_id)

    with open(label_list_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            label_id = row[0]
            label_name = row[1]
            label_dict[label_name] = label_id

    with open(dataset_path, 'r') as csvfile_dataset, \
         open(autotags_path, 'r') as csvfile_autotags, \
         open(metadata_path, 'w+') as csvfile_metadata:

        d_reader = csv.reader(csvfile_dataset, delimiter='\t', quoting=csv.QUOTE_NONE)
        a_reader = csv.reader(csvfile_autotags, delimiter='\t', quoting=csv.QUOTE_NONE)
        m_writer = csv.writer(csvfile_metadata, delimiter=' ')

        for i, d_row in enumerate(d_reader):
            # write a short progress message
            if i % 1e4 == 0:
                print(i)

            if (i + process_id) % num_processes != 0:
                continue

            # if download path ends with '.jpg', ignore it
            if os.path.splitext(d_row[16])[1] != '.jpg':
                continue

            d_id = d_row[1]
            file_name = os.path.join(image_folder, d_id+'.jpg')

            if not os.path.isfile(file_name):
                continue

            print(file_name)

            autotags_found = False
            # iterate over autotags
            for a_row in a_reader:
                a_id = a_row[0]

                # if indices match, extract metadata
                if d_id == a_id:
                    meta_list = []

                    # find all labels
                    for pair in a_row[1].split(','):
                        name = pair.split(':')[0]

                        if len(name) > 0:
                            prob = pair.split(':')[1]
                            meta_list.append((int(label_dict[name]),prob))

                    meta_list.sort()

                    m_writer.writerow(['/images/'+d_id+'.jpg'] + list(chain.from_iterable(meta_list)))
                    autotags_found = True
                    break

            if not autotags_found:
                print('No autotags found: ' + str(file_name))


def merge_metadata(metadata_path):
    split_files = glob.glob(metadata_path + '_*')

    with open(metadata_path,'w+') as wf:
        for f in split_files:
            with open(f,'r') as fd:
                shutil.copyfileobj(fd, wf)


def create_splits(metadata_path, base_folder):
    train_file = os.path.join(base_folder, 'train.txt')
    test_file = os.path.join(base_folder, 'test.txt')

    with open(metadata_path, 'r') as f_metadata, \
         open(train_file, 'w+') as f_train, \
         open(test_file, 'w+') as f_test:

        for i, line in enumerate(f_metadata):
            # write a short progress message
            if i % 1e3 == 0:
                print(i)

            if i % TEST_RATIO == 0:  # write every n-th line to test file
                f_test.write(line)
            else:
                f_train.write(line)





if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(arg)
        create_metadata_parallel(AUTOTAGS, DATASET, IMAGE_FOLDER, LABEL_LIST, METADATA, int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    #else:
    #    download_parallel(DATASET, IMAGE_FOLDER, TEMP_FOLDER, FAILED_FOLDER, NUM_PROCESSES)

    #create_metadata(AUTOTAGS, DATASET, IMAGE_FOLDER, LABEL_LIST, METADATA)

