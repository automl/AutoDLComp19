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
AUTOTAGS = os.path.join(BASE_FOLDER, 'unzip/yfcc100m_autotags')
# path to the original dataset file
DATASET = os.path.join(BASE_FOLDER, 'unzip/yfcc100m_dataset')

# path to the file containing all possible labels as a list
LABEL_LIST = os.path.join(BASE_FOLDER, 'yfcc100m_label')
# number of cpus, determines number of parallel processes when extracting the download links
METADATA = os.path.join(BASE_FOLDER, 'yfcc100m_metadata')
# path to the video download folder
DOWNLOAD_FOLDER = os.path.join(BASE_FOLDER, 'download')
# path to the frame download folder
FRAME_FOLDER = os.path.join(BASE_FOLDER, 'frames')
# path to the frame download folder
DELETE_FRAME_FOLDER = os.path.join(BASE_FOLDER, 'frame')
# folder where the videos for DALI should be placed
DALI_FOLDER = os.path.join(BASE_FOLDER, 'dali')


# guess what
NUM_PROCESSES = 1

# desired width/height of resulting images
WIDTH_DES = 320.0
HEIGHT_DES = 240.0

# minimum duration of the video in s
DURATION_MIN = 0

# test ratio (1/N)
TEST_RATIO = 5

csv.field_size_limit(100000000)




def download_parallel(dataset_path, download_folder, num_processes):
    '''
    does the same as extract_download_links(), but in parallel
    '''
    p_list = []
    for i in range(num_processes):
        p = mp.Process(target=download, args=(dataset_path, download_folder, i, num_processes))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()



def download(dataset_path, download_folder, process_id=0, num_processes=1):
    '''
    extract the download links from the yfcc100m dataset and store them in a file
    '''
    with open(dataset_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)

        # find all labels and store them in label_dict
        for i, row in enumerate(reader):
            # write a short progress message
            if i % 1e5 == 0:
                print(i)

            # if download path ends with '.jpg', ignore it
            if os.path.splitext(row[16])[1] == '.jpg':
                continue

            # use non-conflicting indices for every process
            if (i+process_id)%num_processes != 0:
                continue

            # main step: load video in browser, extract download link and store it
            id = row[1]
            filename = os.path.join(download_folder, id + '.mp4')

            # do not download files twice
            if os.path.isfile(filename):
                print('File already downloaded: ' + filename)
                continue

            try:
                video_url = row[15]
                print('Found video: ' + video_url)
                download_video(video_url, filename)
            except Exception as err:
                print('-----------------')
                print(err)
                print('-----------------')



def save_video_url(writer, index, id, video_url):
    writer.writerow([index, id, video_url])



def download_video(video_url, filename):
    subprocess.call(['youtube-dl', '-q', '-o', filename, '-f', 'iphone_wifi', video_url])
    if os.path.isfile(filename):
        return

    print('second try')
    subprocess.call(['youtube-dl', '-q', '-o', filename, '-f', '288p', video_url])
    if os.path.isfile(filename):
        return

    print('third try')
    subprocess.call(['youtube-dl', '-q', '-o', filename, '-f', '360p', video_url])
    if os.path.isfile(filename):
        return

    print('fourth try')
    subprocess.call(['youtube-dl', '-q', '-o', filename, video_url])
    if os.path.isfile(filename):
        return

    raise Exception('File not downloaded properly: ' + filename)



def get_all_labels(autotags_path, label_list_path):
    '''
    extract all 1570 labels from the autotags and write them to a file together with their index
    '''
    label_dict = {}

    # find all labels and store them in label_dict
    with open(autotags_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for i, row in enumerate(reader):
            # write a short progress message
            if i%1e6 == 0:
                print(i)

            # find all labels
            for pair in row[1].split(','):
                name = pair.split(':')[0]

                if len(name) > 0:
                    label_dict[name] = 0

    # sort all labels by name
    label_list = sorted(list(label_dict.keys()))

    with open(label_list_path, 'w+') as csvfile_write:
        writer = csv.writer(csvfile_write, delimiter='\t')
        for i, item in enumerate(label_list):
            writer.writerow([i, item])


def convert_to_frames_parallel(download_folder, frame_folder, num_processes):
    '''
    does the same as extract_download_links(), but in parallel
    '''
    p_list = []
    for i in range(num_processes):
        p = mp.Process(target=convert_to_frames, args=(download_folder, frame_folder, i, num_processes))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()



def convert_to_frames(download_folder, frame_folder, process_id=0, num_processes=1):
    '''
    convert videos to frame level
    '''
    # recursively parse root_dir
    for i, file_path in enumerate(sorted(glob.iglob(download_folder + "**/**", recursive=True))):
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

                width = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream=width", "-of", "csv=p=0:s=x", file_path])
                height = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream=height", "-of", "csv=p=0:s=x", file_path])
                duration = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream=duration", "-of", "csv=p=0:s=x", file_path])
                width = float(width)
                height = float(height)
                duration = float(duration.decode('utf-8').split('\n')[0])

                if duration < DURATION_MIN:
                    print('video too short: ' + file_name + ' ' + str(duration))
                    continue

                if not os.path.exists(dest_folder_temp):
                    os.mkdir(dest_folder_temp)

                frac = width/height
                frac_des = WIDTH_DES/HEIGHT_DES

                reshape_command = ""
                if frac > frac_des: # image is wider than usual
                    if height > HEIGHT_DES: # reduce image size
                        reshape_command = "-vf scale=-1:" + str(int(height))
                else: # image is higher than usual
                    if width > WIDTH_DES:   # reduce image size
                        reshape_command = "-vf scale=" + str(int(width)) + ":-1"

                dest = os.path.join(dest_folder_temp, "%04d.jpg")
                command = "ffmpeg -v error -i " + file_path + " -y -r 5 " + reshape_command + " " + dest
                os.system(command)
                os.rename(dest_folder_temp, dest_folder)
            except Exception as err:
                print('-----------------')
                print(err)
                if os.path.exists(dest_folder_temp):
                    shutil.rmtree(dest_folder_temp)
                print('-----------------')



def create_metadata(autotags_path, dataset_path, frame_folder, label_list_path, metadata_path):
    '''
    create metadata based on downloaded files
    '''
    label_dict = {}

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
            if i % 1e6 == 0:
                print(i)

            # if download path ends with '.jpg', ignore it
            if os.path.splitext(d_row[16])[1] == '.jpg':
                continue

            d_id = d_row[1]
            folder_name = os.path.join(frame_folder, d_id)

            if not os.path.isdir(folder_name):
                continue

            num_frames = len(next(os.walk(folder_name))[2])

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

                    m_writer.writerow(['/frames/'+d_id+'/', str(num_frames)] + list(chain.from_iterable(meta_list)))
                    autotags_found = True
                    break

            if not autotags_found:
                print('No autotags found: ' + str(folder_name))



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



def delete_frame_folder_parallel(metadata_path, delete_frame_folder, num_processes):
    '''
    does the same as extract_download_links(), but in parallel
    '''
    p_list = []
    for i in range(num_processes):
        p = mp.Process(target=delete_frame_folder, args=(metadata_path, delete_frame_folder, i, num_processes))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()



def delete_frame_folder(metadata_path, delete_frame_folder, process_id, num_processes):
    with open(metadata_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE)
        for i, row in enumerate(reader):
            if i % 1e4 == 0:
                print(i)

            # use non-conflicting indices for every process
            if (i+process_id)%num_processes != 0:
                continue

            video_id = row[0].split('/')[2]
            delete_folder = os.path.join(delete_frame_folder, video_id)

            if os.path.isdir(delete_folder):
                shutil.rmtree(delete_folder)
                print('deleted folder: ' + delete_folder)


def dali_parallel(download_folder, dali_folder, dataset_path, process_start, process_end, num_processes):
    p_list = []
    for i in range(process_start, process_end):
        p = mp.Process(
            target=dali,
            args=(download_folder, dali_folder, dataset_path, i, num_processes
            )
        )
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()


def dali(download_folder, dali_folder, dataset_path, process_id=0, num_processes=1):
    if not os.path.isdir(dali_folder):
        os.mkdir(dali_folder)

    with open(dataset_path, 'r') as csvfile_dataset:
        d_reader = csv.reader(csvfile_dataset, delimiter='\t', quoting=csv.QUOTE_NONE)

        for i, d_row in enumerate(d_reader):
            # write a short progress message
            if i % 1e5 == 0:
                print(i)

            if (i + process_id) % num_processes != 0:
                continue

            video_id = d_row[1]
            file_path = os.path.join(download_folder, str(video_id)+'.mp4')

            if not os.path.isfile(file_path):
                continue

            print(file_path)

            video_folder_new = os.path.join(dali_folder, video_id)
            file_path_new = os.path.join(video_folder_new, str(video_id)+'.mp4')

            if not os.path.isdir(video_folder_new):
                os.mkdir(video_folder_new)

            shutil.move(file_path, file_path_new)



def dali_replace_paths(metadata_path):
    output_path_new = metadata_path + '_new'
    with open(metadata_path, 'r') as csvfile, \
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
        #convert_to_frames(DOWNLOAD_FOLDER, FRAME_FOLDER, int(sys.argv[1]), int(sys.argv[2]))
        #delete_frame_folder(METADATA, DELETE_FRAME_FOLDER, int(sys.argv[1]), int(sys.argv[2]))
        #dali_parallel(DOWNLOAD_FOLDER, DALI_FOLDER, DATASET, int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        pass
    else:
        #get_all_labels(AUTOTAGS, LABEL_LIST)
        #download_parallel(DATASET, DOWNLOAD_FOLDER, NUM_PROCESSES)
        #convert_to_frames_parallel(DOWNLOAD_FOLDER, FRAME_FOLDER, NUM_PROCESSES)
        #create_metadata(AUTOTAGS, DATASET, FRAME_FOLDER, LABEL_LIST, METADATA)
        #create_splits(METADATA, BASE_FOLDER)
        create_splits(METADATA, BASE_FOLDER)

