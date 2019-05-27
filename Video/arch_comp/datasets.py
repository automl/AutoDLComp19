import os
import cv2
import csv
import numpy as np
import numpy.matlib as matl
import scipy.io
import torch.utils.data as data
import eioh


class Dataset():
    def __init__(self, dataset='ucf101', data_dir="."):
        if dataset == 'ucf101':
            self.train_loader = data.DataLoader(dataset=UCF101(data_dir=data_dir, type='train'), batch_size=1, shuffle=True)
            self.valid_loader = data.DataLoader(dataset=UCF101(data_dir=data_dir, type='valid'), batch_size=1, shuffle=True)
            self.test_loader = data.DataLoader(dataset=UCF101(data_dir=data_dir, type='test'), batch_size=1, shuffle=False)
        elif dataset == 'charades':
            pass
            #self.dataset = Charades(data_dir)
        else:
            raise Exception('dataset not handled: ' + str(dataset))

    def get_data_loader(self):
        return self.train_loader, self.valid_loader, self.test_loader


class UCF101(data.Dataset):
    '''
    Data loader for the ucf 101 dataset. It assumes that in the top-level folder there is another
    folder (aaa_recognition in our case) with the files for the test/train split
    '''
    def __init__(self, data_dir, type):
        self.data_dir = data_dir

        # path to class index file
        index_path = os.path.join(os.path.abspath(data_dir), 'aaa_recognition/classInd.txt')
        train_list_path = os.path.join(os.path.abspath(data_dir), 'aaa_recognition/trainlist01.txt')
        test_list_path = os.path.join(os.path.abspath(data_dir), 'aaa_recognition/testlist01.txt')

        train_dataset, valid_dataset = self.parse_train_list(train_list_path)
        test_dataset, self.num_classes = self.parse_test_list(index_path, test_list_path)

        if type == 'train':
            self.dataset = train_dataset
        elif type == 'valid':
            self.dataset = valid_dataset
        elif type == 'test':
            self.dataset = test_dataset
        else:
            raise Exception('Unknown dataset type: ' + str(type))


    def parse_train_list(self, train_list_path):
        train_dataset = []
        valid_dataset = []

        # load train dataset
        with open(train_list_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                abs_path = os.path.join(os.path.abspath(self.data_dir), row[0])
                class_label = int(row[1])
                train_dataset.append((abs_path, class_label))

        # assign one third of the training data to the validation data
        for i in range(int(len(train_dataset)/3)):
            valid_dataset.append(train_dataset.pop(0))

        return train_dataset, valid_dataset


    def parse_test_list(self, index_path, test_list_path):
        index_list = {}
        num_classes = 0

        # load mapping between names and labels
        with open(index_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                class_name = row[1]
                class_label = int((row[0]))
                num_classes = max(class_label, num_classes)
                index_list[class_name] = class_label

        test_dataset = []

        # load test dataset and assign correct labels
        with open(test_list_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                abs_path = os.path.join(os.path.abspath(self.data_dir), row[0])
                class_name = row[0].split('/')[0]
                class_label = index_list[class_name]

                test_dataset.append((class_name, class_label))

        return test_dataset, num_classes


    def __getitem__(self, idx):
        '''
        load video from disk and return it with corresponding label

        :param idx: index of the video in paths_and_labels
        :return: array (video,label)
        '''
        path = self.dataset[idx][0]
        lbl = self.dataset[idx][1]

        # one-hot-encoding of label
        label = np.zeros([self.num_classes])
        label[lbl-1] = 1

        if os.path.isfile(path):
            video = eioh.load_video_file(path)
        elif os.path.isdir(path):
            video = eioh.load_image_dir(path)
        else:
            raise Exception('unknown path type')
        label = matl.repmat(label,video.shape[0],1)
        return (video,label)


# class Charades(data.dataset):
#     '''
#     Data loader for the ucf 101 dataset. It assumes that in the top-level folder there is another
#     folder (aaa_recognition in our case) with the files for the test/train split
#     '''
#     def __init__(self, data_dir):
#         pass


if __name__ == "__main__":
    ucf = Dataset(dataset='ucf101', data_dir = '/home/dingsda/autodl/data/ucf101')

