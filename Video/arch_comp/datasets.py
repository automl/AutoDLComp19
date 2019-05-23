import os
import cv2
import numpy as np
import numpy.matlib as matl
import scipy.io
import torch.utils.data as data
import eioh


class Dataset():
    def __init__(self, dataset='ucf101', data_dir=".", label_path="./labels.mat", split=[0.6, 0.2, 0.2]):
        if dataset == 'ucf101':
            self.dataset = UCF101(data_dir, label_path)
            self._init_data_loader(split)
        else:
            raise Exception('dataset not handled: ' + str(dataset))


    def _init_data_loader(self, split):
        train_idx, valid_idx, test_idx = self._get_split(split)

        train_sampler = data.SubsetRandomSampler(train_idx)
        valid_sampler = data.SubsetRandomSampler(valid_idx)
        test_sampler = data.SubsetRandomSampler(test_idx)


        self.train_loader = data.DataLoader(dataset=self.dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            sampler=train_sampler)
        self.valid_loader = data.DataLoader(dataset=self.dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            sampler=valid_sampler)
        self.test_loader = data.DataLoader(dataset=self.dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           sampler=test_sampler)

    def _get_split(self, split):
        '''
        calculate train/validation/test splits

        :return: indices of dataset belonging to the specific split
        '''

        l = self.dataset.len()
        idx = np.arange(l)
        np.random.seed(1337)
        np.random.shuffle(idx)

        split = np.array(split)
        split = split / sum(split)
        split = split.cumsum()*l
        split = split.astype(int)

        train_idx = idx[:split[0]]
        valid_idx = idx[split[0]:split[1]]
        test_idx = idx[split[1]:]

        return train_idx, valid_idx, test_idx


    def get_label_size(self):
        return self.dataset.get_label_size()


    def get_data_loader(self):
        return self.train_loader, self.valid_loader, self.test_loader


class UCF101(data.Dataset):
    def __init__(self, data_dir=".", label_path="./labels.mat"):

        paths = self._parse_paths(data_dir)
        labels = self._parse_labels(label_path)
        self.metadata = self._merge_paths_and_labels(paths,labels)


    def _parse_paths(self, data_dir):
        '''
        get list of all absolute UCF101 video file paths

        Note: Because the folder names do not correspond 1:1 to the labels in the
        corresponding label file, we cannot use the folder names to identify
        the class of each video. Rather we have to sort them alphabetically and
        hope for the best :)

        :param data_dir: path to the UCF101 folder containing all videos
        :return: 2D list of all UCF101 file paths. First dimension corresponds to class
        '''

        paths = []

        for dir_name in sorted(os.listdir(data_dir)):
            abs_dir_name = os.path.join(os.path.abspath(data_dir), dir_name)

            if os.path.isdir(abs_dir_name):
                dir_paths = []

                for video_name in os.listdir(abs_dir_name):
                    abs_video_name = os.path.join(os.path.abspath(abs_dir_name), video_name)

                    if os.path.isfile(abs_video_name) or os.path.isdir(abs_video_name):
                        dir_paths.append(abs_video_name)
                    else:
                        raise Exception('directory should only contain files or folders: ' + abs_dir_name)
                paths.append(dir_paths)
            else:
                raise Exception('no directory: ' + abs_dir_name)

        if len(paths) != 101:
            raise Exception('not exactly 101 classes in dataset')

        return paths


    def _parse_labels(self, label_path):
        '''
        get the labels for all UCF101 classes

        :param label_path: path to the label file
        :return: numpy array with shape 101(classes)x115(labels)
        '''

        mat = scipy.io.loadmat(label_path)
        # mapping between classes and labels
        vals = mat['class_attributes'][0][0][2]
        vals = np.array(vals).transpose().astype(float)

        if vals.shape[0] != 101:
            raise Exception('not exactly 101 classes in label file')

        return vals


    def _merge_paths_and_labels(self, files, labels):
        '''
        :param paths: list of all UCF101 videos
        :param labels: array of all UCF101 labels
        :return: list of corresponding videos/labels
        '''

        paths_and_labels = []
        for i in range(len(files)):
            for file in files[i]:
                paths_and_labels.append((file,labels[i]))

        return paths_and_labels


    def __len__(self):
        return len(self.metadata)


    def len(self):
        return len(self.metadata)


    def get_label_size(self):
        return len(self.metadata[0][1])


    def __getitem__(self, idx):
        '''
        load video from disk and return it with corresponding label

        :param idx: index of the video in paths_and_labels
        :return: array (video,label)
        '''
        path = self.metadata[idx][0]
        label = self.metadata[idx][1]

        if os.path.isfile(path):
            video = eioh.load_video_file(path)
        elif os.path.isdir(path):
            video = eioh.load_image_dir(path)
        else:
            raise Exception('unknown path type')
        label = matl.repmat(label,video.shape[0],1)
        return (video,label)


if __name__ == "__main__":
    ucf = Dataset(dataset = 'ucf101',
                  data_dir = '../data/ucf101_frames',
                  label_path = '../data/class_attributes_UCF101.mat')
    print(ucf.dataset.metadata)

