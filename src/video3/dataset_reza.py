import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import torch
import csv
from collections.abc import Iterable
from numpy.random import randint


def convert_path_to_id(path):
    if path.endswith('.mp4'):
        return path.split('/')[-1][:-4]
    elif path.endswith('/'):
        return path.split('/')[-2]


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def labels(self):
        return self._data[2:]


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file, additional_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False,
                 classification_type='multiclass', num_labels=500):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.classification_type = classification_type

        if self.classification_type == 'multilabel':
            self.label_limiter = LabelLimiter(list_file, additional_file, des_num_labels=num_labels)

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            # x_img = x_img.cuda()
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]  # noqa: E501
        self.video_list = []
        for x in open(self.list_file):
            data = x.strip().split(' ')
            path = '{}{}'.format(self.root_path, data[0]).replace('//', '/')
            self.video_list.append(VideoRecord([path] + data[1:]))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                seg_imgs = seg_imgs
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)

        if self.classification_type == 'multiclass':
            label = int(record.labels[0])
        elif self.classification_type == 'multilabel':
            id = convert_path_to_id(record.path)
            label = self.label_limiter.get(id)
        else:
            raise NotImplementedError('unknown classification type: ' + str(self.classification_type))

        return process_data, label

    def __len__(self):
        return len(self.video_list)



class LabelLimiter():
    def __init__(self, list_file, additional_file, des_num_labels=100):
        print('loading dataset for label limiter')
        data = {}
        data.update(self._load_file(list_file))
        data.update(self._load_file(additional_file))
        index, self.num_labels = self._get_most_occ_labels(data, des_num_labels)
        self.data = self._realign_labels(data, index)

    def get(self, ids):
        '''
        get the labels as a torch tensor for either a single video id or a list of them
        '''
        if isinstance(ids, Iterable) and not isinstance(ids, str):
            label_tensor = torch.zeros([len(ids), self.num_labels])
            for i, id in enumerate(ids):
                label_tensor[i] = self._get_single_id(id)
        else:
            label_tensor = self._get_single_id(ids)

        return label_tensor


    def _get_single_id(self, id):
        '''
        get the labels as a torch tensor for a single video id (only internally called)
        '''
        label_tensor = torch.zeros(self.num_labels)
        if id in self.data:
            val = self.data[id]
            for idx, perc in zip(val[::2], val[1::2]):
                label_tensor[idx] = perc
            return label_tensor
        else:
            raise IndexError('unknown video id: ' + str(id))



    def _load_file(self, file):
        '''
        load file
        '''
        data = {}

        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                id = convert_path_to_id(row[0])
                data[id] = [int(elem) if i%2==0 else float(elem) for i, elem in enumerate(row[2:])]

        return data


    def _get_num_labels(self, data):
        '''
        get total number of different labels
        '''
        num_labels = 0
        for _,val in data.items():
            if val:
                num_labels = max(num_labels, max(val))
        return num_labels+1


    def _get_most_occ_labels(self, data, des_num_labels):
        '''
        get the x most occurring labels in aascending order
        '''
        num_labels = self._get_num_labels(data)

        if des_num_labels > num_labels:
            return np.array(range(num_labels)), num_labels

        freq = np.zeros(num_labels)
        for _,val in data.items():
            if val:
                freq[val[::2]] += val[1::2]

        index = np.argsort(-freq)[:des_num_labels]
        return np.sort(index), des_num_labels


    def _realign_labels(self, data, index):
        '''
        realign the labels to be limited to the indices stored in 'index'
        '''
        index_dict = {}
        for i, val in enumerate(index):
            index_dict[val] = i

        for key, val in data.items():
            val_new = []
            if val:
                for idx, perc in zip(val[::2], val[1::2]):
                    if idx in index_dict:
                        val_new.append(index_dict[idx])
                        val_new.append(perc)
            data[key] = val_new

        return data
