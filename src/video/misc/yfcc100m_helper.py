import csv
import os
import wget
import multiprocessing as mp
import urllib.parse
import requests
import re


# path to the original autotag file
AUTOTAGS = '/home/dingsda/Downloads/yfcc100m_autotags'
# path to the original dataset file
DATASET = '/media/dingsda/External/datasets/yfcc100m/yfcc100m_dataset'

# path to the file containing all possible labels as a list
LABEL_LIST = '/home/dingsda/autodl/data/yfcc100m/yfcc100m_label'
# path to the file containing all video download links as a list
DOWNLOAD_LIST = '/home/dingsda/autodl/data/yfcc100m/yfcc100m_download'
# number of cpus, determines number of parallel processes when extracting the download links
METADATA = '/home/dingsda/autodl/data/yfcc100m/yfcc100m_metadata'
# path to the video download folder
DOWNLOAD_FOLDER = '/home/dingsda/autodl/data/yfcc100m'

# guess what
NUM_PROCESSES = 64

csv.field_size_limit(100000000)


def get_download_links_parallel(dataset_path, download_list_path, num_processes):
    '''
    does the same as extract_download_links(), but in parallel
    '''
    print('Number of CPUs: ' + str(num_processes))

    p_list = []
    for i in range(num_processes):
        p = mp.Process(target=get_download_links, args=(dataset_path, download_list_path, i, num_processes))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()



def get_download_links(dataset_path, download_list_path, process_id=0, num_processes=1):
    '''
    extract the download links from the yfcc100m dataset and store them in a file
    '''
    download_list_path = download_list_path + str(process_id)

    process_active = True
    id_last = ''

    # load file containing all download links and find id of last processed video
    if os.path.isfile(download_list_path):
        with open(download_list_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                id_last = row[1]
                process_active = False

    with open(download_list_path, 'a+') as csvfile_write, \
         open(dataset_path, 'r') as csvfile:
        writer = csv.writer(csvfile_write, delimiter='\t')
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
            index = row[0]
            id = row[1]
            if process_active == True:
                video_url = row[15]
                t_wait = 1
                print('Found video: ' + video_url)

                # consider timeouts when loading the page, thus try again with increased timeout
                try:
                    save_download_url(writer, video_url, index, id)
                except Exception as err:
                    print(err)

            # trigger processing
            if id == id_last:
                process_active = True



def save_download_url(writer, video_url, index, id):
    '''
    small helper function. Open URL and extract download URL from HTML tag
    '''
    query = 'https://qdownloader.net/download?video=' + urllib.parse.quote_plus(video_url)
    r = requests.get(query)
    x = re.search(r"\"http\S+orig\S+\"", r.text)

    if x:
        download_url = x.group()[1:-1]
        writer.writerow([index, id, download_url])



def merge_download_links(download_list_path, num_processes):
    '''
    merge multiple files containing download links into a single file
    '''
    download_list = []

    # copy download links from individual files
    for i in range(num_processes):
        file_path = download_list_path + str(i)
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')

            for row in reader:
                download_list.append((row[0], row[1], row[2]))

    # sort ist in ascending id order. Important for get_metadata()
    download_list.sort(key = lambda x: int(x[0]))

    # write combined download links to sinle file
    with open(download_list_path, 'w+') as csvfile_write:
        writer = csv.writer(csvfile_write, delimiter='\t')

        for elem in download_list:
            index = elem[0]
            id = elem[1]
            download_url = elem[2]
            writer.writerow([index, id, download_url])



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



def get_metadata(download_path, autotags_path, label_list_path, metadata_path):
    '''
    merge all information into a metadata file
    '''
    label_dict = {}

    with open(label_list_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            label_id = row[0]
            label_name = row[1]
            label_dict[label_name] = label_id

    with open(download_path, 'r') as csvfile_download, \
         open(autotags_path, 'r') as csvfile_autotags, \
         open(metadata_path, 'w+') as csvfile_metadata:
        reader_download = csv.reader(csvfile_download, delimiter='\t')
        reader_autotags = csv.reader(csvfile_autotags, delimiter='\t')
        writer_metadata = csv.writer(csvfile_metadata, delimiter='\t')

        # iterate over found videos
        for d_row in reader_download:
            d_id = d_row[1]
            meta_list = []

            # iterate over autotags
            for a_row in reader_autotags:
                a_id = a_row[0]

                # if indices match, extract metadata
                if d_id == a_id:
                    # find all labels
                    for pair in a_row[1].split(','):
                        name = pair.split(':')[0]

                        if len(name) > 0:
                            prob = pair.split(':')[1]
                            meta_list.append(label_dict[name])
                            meta_list.append(prob)
                    writer_metadata.writerow([d_id] + meta_list)
                    break


def download_videos(download_list_path, download_folder):
    '''
    download all files stored in the download list file
    '''

    with open(download_list_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            id = row[1]
            download_url = row[2]
            filename = wget.download(download_url)
            filename_target = os.path.join(download_folder, id+'.mp4')
            os.rename(filename, filename_target)
            print('Downloaded ' + filename_target)


if __name__ == "__main__":
    #get_all_labels(AUTOTAGS, LABEL_LIST)
    get_download_links_parallel(DATASET, DOWNLOAD_LIST, NUM_PROCESSES)
    #get_download_links(DATASET, DOWNLOAD_LIST)
    #merge_download_links(DOWNLOAD_LIST, NUM_CPU)
    #get_metadata(DOWNLOAD_LIST, AUTOTAGS, LABEL_LIST, METADATA)
    #download_videos(DOWNLOAD_LIST, DOWNLOAD_FOLDER)





















