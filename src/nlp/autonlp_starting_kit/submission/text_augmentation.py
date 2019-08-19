import numpy as np

# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)

class Augmentation():
    def __init__(self, imbalance, max_size=100000, threshold=0.1):
        self.imbalance = imbalance
        self.max_size = max_size
        self.threshold = threshold
        self.classes = len(imbalance)

    def augment_data(self, data, labels):
        labels_ohe = ohe2cat(labels)
        imbalance = np.array(self.imbalance)
        aug_labels = np.where(imbalance > self.threshold)[0]
        if len(aug_labels) == 0:
            return data, labels
        aug_data_size = np.ceil(np.array(imbalance)[aug_labels] * len(data)).astype(int)
        aug_data_x = []
        aug_data_y = []
        for i, label in enumerate(aug_labels):
            req_size = min(aug_data_size[i], self.max_size)
            label_idx = np.where(labels_ohe == label)[0]
            for j in np.random.choice(range(len(label_idx)), req_size, replace=True):
                tokens = data[label_idx[j]]
                if np.random.uniform() > 0.5:
                    tokens = self._aug_reverse_tokens(tokens)
                else:
                    tokens = self._aug_random_deletion(tokens)
                aug_data_x.append(tokens)
                y = np.zeros(self.classes).tolist()
                y[label] = 1
                aug_data_y.append(y)
        data.extend(aug_data_x)
        labels = np.vstack((labels, aug_data_y))
        return data, labels

    def _reconstruct(self, tokens, start, end):
        tokens.insert(0, start)
        tokens.insert(len(tokens), end)
        return tokens

    def _aug_reverse_tokens(self, tokens):
        '''Reverses a list of tokens keeping the [CLS] and [SEP] intact
        '''
        start = tokens[0]
        end = tokens[-1]
        tokens = np.flip(tokens[1:-1]).tolist()
        return self._reconstruct(tokens, start, end)

    def _aug_random_deletion(self, tokens, max_size=3):
        ''' Randomly deletes words

        Parameters
        ----------
        tokens : list
        max_size : int
            At most len(tokens) / max_size words can be deleted
            If len(tokens) < max_size, no token will be deleted

        Returns
        -------
        list
        '''
        start = tokens[0]
        end = tokens[-1]
        tokens = np.array(tokens[1:-1])
        length = len(tokens)
        max_lim = int(length / max_size)
        low_lim = min(max_size, max_lim)
        if low_lim >= max_lim:
            return self._reconstruct(tokens.tolist(), start, end)
        delete_count = np.random.randint(low_lim, max_lim)
        delete_index = np.random.choice(range(length), delete_count, replace=False)
        tokens = np.delete(tokens, delete_index).tolist()
        return self._reconstruct(tokens, start, end)
