import csv
import numpy as np
import torch
from collections.abc import Iterable

class Dali():
    def __init__(self, train_file, test_file, des_num_labels):
        self.data, self.num_labels = self._limit_labels(train_file, test_file, des_num_labels)


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
                data[row[0]] = [int(elem)-1 if i%2==0 else float(elem) for i, elem in enumerate(row[2:])]

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


    def _limit_labels(self, train_file, test_file, des_num_labels):
        '''
        do not use all labels but only a subset of the most occurring ones
        '''
        data = {}
        data.update(self._load_file(train_file))
        data.update(self._load_file(test_file))
        index, num_labels = self._get_most_occ_labels(data, des_num_labels)
        data = self._realign_labels(data, index)

        return data, num_labels




if __name__ == "__main__":
    dali = Dali('/home/dingsda/train.txt', '/home/dingsda/test.txt', 10)
    print(dali.get(['408522', '423566']))
    print(dali.get('592965'))