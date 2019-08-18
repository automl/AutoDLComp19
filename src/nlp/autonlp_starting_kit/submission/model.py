# -*- coding: utf-8 -*-

import pytorch_transformers as pytrf
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

import re

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split

from scoring import autodl_auc
from bert import BertTokenizer, NLPBertClassifier, BERT_PRETRAINED
from xlnet import XLNetTokenizer, NLPXLNetClassifier, XLNET_PRETRAINED

SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MAX_STR_LEN = 2500
MAX_TOK_LEN = 512


# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)


def bucket_shuffle(arr, bucket_size, pad_key=-1):
    ''' shuffle 1-D array in buckets'''

    # pad so that length divisible by bucket size
    pad_len = len(arr) % bucket_size
    pad_len = 0 if pad_len == 0 else bucket_size - pad_len
    arr = np.pad(arr, pad_width=(0, pad_len), mode='constant', constant_values=(pad_key, pad_key))
    # shuffle within bucket
    arr = arr.reshape(-1, bucket_size)
    arr = np.apply_along_axis(np.random.permutation, 1, arr).flatten()
    # remove padding key and return
    return arr[arr != pad_key]


class TextLoader():
    def __init__(self, data, label, metadata, tokenizer, batch_size=1, shuffle=False, sort=False,
                 device=torch.device('cpu'), workers=4):
        """
        tokenizes dataset with given tokenizer and generates batches of data
        data: list of sequences
        label: list of labels
        metadata: dict
        tokenizer: class implementing AbstractTokenizer
            needs to have method 'tokenize' that takes text/sentences and returns tokenized ids
        shuffle: to shuffle the data
        sort: to sort the sequences by length
            would lead to sequences of equal size together
        device: torch device
        workers: parallel workers for tokenization
        """

        self.device = device
        # tokenize sentences
        # data = np.array([tokenizer.tokenize(d) for d in data])
        data = tokenizer.tokenize(data, max_str_len=min(MAX_STR_LEN, metadata['train_cutoff_len']), workers=workers)
        self.data = np.array(data)
        self.label = label
        self.pad_token = 0
        self.data_size = len(self.data)

        self.sort = sort
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.indices = np.arange(self.data_size)

        self.text_lengths = np.array([[i, len(d)] for i, d in enumerate(self.data)])

        if self.sort:
            # sorted indices used for batching
            self.indices = self.text_lengths[self.text_lengths[:, 1].argsort()][:, 0]

    def _generate_batch(self, indices):
        """ generate batch using given indices with padding """
        text = self.data[indices]
        # pad batch to max batch len
        max_batch_len = max(self.text_lengths[indices, 1])
        text = [b + [self.pad_token] * (max_batch_len - len(b)) for b in text]
        # convert to tensor
        text = torch.tensor(text).to(self.device).long()
        # attention mask: 1 - unmasked, 0 - masked
        mask = ~text.eq(self.pad_token)
        if self.label is not None:
            label = self.label[indices]
            label = torch.tensor(label).to(self.device).long()
            return text, label, mask
        return text, None, mask

    def __iter__(self):
        """ generate iterator for padding """

        if self.sort and self.shuffle:
            # if sorted shuffle, then shuffle only within the buckets
            self.indices = bucket_shuffle(self.indices, bucket_size=self.batch_size + 1)
        elif self.shuffle:
            # shuffle dataset completely
            np.random.shuffle(self.indices)

        for i in range(0, self.data_size, self.batch_size):
            if i <= self.data_size - self.batch_size:
                indices = self.indices[i:(i + self.batch_size)]
            else:
                indices = self.indices[i:self.data_size]
            yield self._generate_batch(indices)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))


class NaiveModel():
    def __init__(self, metadata, features, classifier=None):
        self.metadata = metadata
        self.classes = metadata['class_num']
        self.features = features
        self.hv = HashingVectorizer(n_features=features)
        if classifier is not None:
            if classifier == 'lr':
                self.clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
            else:
                self.clf = AdaBoostClassifier(n_estimators=25, learning_rate=1)
        else:
            # # Run logistic (multinomial) regression for all inputs except when
            # # 1) class imbalance is more than 0.2 and
            # # 2) number of training samples are more than 80000
            if max(self.metadata['imbalance']) > 0.2 or self.metadata['train_num'] < 80000:
                self.clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
            else:
                self.clf = AdaBoostClassifier(n_estimators=25, learning_rate=1)

    def _transform(self, data):
        return self.hv.transform(data)

    def train(self, data):
        data_x, data_y = data
        data_y = ohe2cat(data_y)
        data_x = self._transform(data_x)
        self.clf.fit(data_x, data_y)

    def test(self, data):
        data = self._transform(data)
        preds = self.clf.predict(data)
        y_test = np.zeros([preds.shape[0], self.metadata['class_num']])
        for idx, y in enumerate(preds):
            y_test[idx][y] = 1
        return y_test


class Preprocess(object):
    def __init__(self, metadata):
        self.metadata = metadata
        self.classes = metadata['class_num']

        self.REPLACE_BY_SPACE_EN = re.compile('["/(){}\[\]\|@,;]')
        self.REPLACE_BY_SPACE_ZH = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')
        self.BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))

    def preprocess_text(self, data, cutoff=75):
        '''Cleans/preprocesses a list of list of English/Chinese strings

        Parameters
        ----------
        data : tuple
          (List of training strings, List of one-hot training labels)
        cutoff : the percentile of string length to find in data_x

        Returns
        -------
        tuple, (List of cleaned training strings, List of one-hot labels)
        '''
        data_x, data_y = data
        clean_text = {'EN': self._clean_en_text, 'ZH': self._clean_zh_text}
        clean_text = clean_text[self.metadata['language']]
        data_x, _, _ = clean_text(data_x, data_y, cutoff)
        return (data_x, data_y)

    def _clean_en_text(self, data_x, data_y, cutoff=90, max_str_len=MAX_STR_LEN):
        '''Cleans/preprocesses a list of list of English strings

        Parameters
        ----------
        data_x : list of list
          Contains the list of input strings
        data_y : list of list
          Contains the list of one-hot-encoded class labels
        cutoff : the percentile of string length to find in data_x

        Returns
        -------
        ret : list of list
          Contains list of cleaned strings
        class_freq : list
          List of class frequencies in data
        cutoff_len : float
          'cutoff' percentile of string length in data
        '''

        if data_y is None:
            data_y = [0 for i in range(len(data_x))]

        ret = []
        class_freq = [0 for i in range(self.classes)]
        len_list = []
        for i, line in enumerate(data_x):
            line = self.convert_to_unicode(line)
            # line = line.decode('utf-8', 'ignore')
            line = line.lower()  # lowercase text
            line = self.REPLACE_BY_SPACE_EN.sub(' ', line)
            line = self.BAD_SYMBOLS_RE.sub('', line)
            line = line.strip()
            len_list.append(len(line))
            ret.append(line)
            class_freq[int(np.argmax(data_y[i]))] += 1
        cutoff_len = np.percentile(len_list, cutoff)
        if data_y is not None:
            self.metadata['class_freq'] = class_freq
            self.metadata['train_cutoff_len'] = int(cutoff_len)
            self._imbalance_stats()
            cutoff_len = class_freq = None
        return ret, class_freq, cutoff_len

    def _clean_zh_text(self, data_x, data_y, cutoff=90, max_str_len=MAX_STR_LEN):
        '''Cleans/preprocesses a list of list of Chinese strings

        Parameters
        ----------
        data_x : list of list
          Contains the list of input strings
        data_y : list of list
          Contains the list of one-hot-encoded class labels
        cutoff : the percentile of string length to find in data_x

        Returns
        -------
        ret : list of list
          Contains list of cleaned strings
        class_freq : list
          List of class frequencies in data
        cutoff_len : float
          'cutoff' percentile of string length in data
        '''

        if data_y is None:
            data_y = [0 for i in range(len(data_x))]

        ret = []
        class_freq = [0 for i in range(self.classes)]
        len_list = []
        for i, line in enumerate(data_x):
            line = self.convert_to_unicode(line)
            # line = line.decode('utf-8', 'ignore')
            line = self.REPLACE_BY_SPACE_ZH.sub(' ', line)
            line = line.strip()
            len_list.append(len(line))
            ret.append(line)
            class_freq[int(np.argmax(data_y[i]))] += 1
        cutoff_len = np.percentile(len_list, cutoff)
        if data_y is not None:
            self.metadata['class_freq'] = class_freq
            self.metadata['train_cutoff_len'] = int(cutoff_len)
            self._imbalance_stats()
            cutoff_len = class_freq = None
        return ret, class_freq, cutoff_len

    def _imbalance_stats(self):
        expected = self.metadata['train_num'] / self.metadata['class_num']
        imbalance = []
        for i, val in enumerate(self.metadata['class_freq']):
            imbalance.append(max(0, expected - val) / self.metadata['train_num'])
        self.metadata['imbalance'] = imbalance


class Model(object):
    """
    model of BERT baseline without pretraining
    """

    def __init__(self, metadata, train_output_path="./", test_input_path="./", config=None, split_ratio=0.95):
        """ Initialization for model
        :param metadata: a dict
        """

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.device = torch.device('cpu')

        self.done_training = False
        self.metadata = metadata
        self.train_output_path = train_output_path
        self.test_input_path = test_input_path

        self.preprocess = Preprocess(metadata)

        self.preprocessed_train_dataset = None
        self.preprocessed_test_dataset = None
        self.naive_preds = None

        # Run tracker
        self.run_count = 0
        self.epochs = 0

        # Naive model parameters


        ## Parameters
        # TODO smarter params
        self.train_batches = 1.0  # prob of executing a batch during an epoch
        self.batch_alpha = 1.2  # multiplier to update batch training probability
        self.train_epochs = np.inf  # num of epochs to train before done_training=True
        self.naive_limit = 2  # num of train runs before testing using network
        self.test_runtime = None  # time taken for inference of test_dataset
        self.workers = 1

        # to split train into train & validation
        self.split_ratio = split_ratio if split_ratio else 1.0
        self.score_fn = autodl_auc

        # self.pretrained_path = '/content/'
        self.pretrained_path = os.path.join(os.path.dirname(__file__), 'pretrained/')

        print(config)
        if config is None:
            # setting default config if not provided
            config = {'classifier_layers': 2, 'classifier_units': 256, 'learning_rate': 0.001,
                      'batch_size': 64, 'str_cutoff': 90, 'features': 2000, 'weight_decay': 0.01, 'stop_count': 5,
                      'classifier': None, 'transformer': 'bert', 'layers': 2, 'optimizer': 'adamw'}

        self.classifier_layers = config['classifier_layers']
        self.classifier_units = config['classifier_units']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.str_cutoff = config['str_cutoff']
        self.features = config['features']
        self.weight_decay = config['weight_decay']
        self.stop_count = config['stop_count']
        self.classifier = config['classifier']
        self.transformer = config['transformer'] if metadata['language'] == 'EN' else 'bert'  # for chinese/english

        # initialize model & tokenizer
        if self.transformer == 'bert':
            self.metadata_pretrain = BERT_PRETRAINED
            self.metadata_pretrain['layers'] = config['layers']
            self.tokenizer = BertTokenizer(self.metadata['language'], self.pretrained_path)
            self.model = NLPBertClassifier(metadata, bert_metadata=self.metadata_pretrain,
                                           pretrained=self.pretrained_path,
                                           classifier_layers=self.classifier_layers,
                                           classifier_units=self.classifier_units)
        elif self.transformer == 'xlnet':
            self.metadata_pretrain = XLNET_PRETRAINED
            self.metadata_pretrain['layers'] = config['layers']
            self.tokenizer = XLNetTokenizer(self.metadata['language'], self.pretrained_path)
            self.model = NLPXLNetClassifier(metadata, xlnet_metadata=self.metadata_pretrain,
                                           pretrained=self.pretrained_path,
                                           classifier_layers=self.classifier_layers,
                                           classifier_units=self.classifier_units)
        self.model.to(self.device)

        if config['optimizer'] == "adamw":
            self.optimizer = pytrf.AdamW(self.model.parameters(), lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.criterion = nn.BCEWithLogitsLoss() if self.metadata['class_num'] == 2 \
            else nn.CrossEntropyLoss()

        # to store train dataset and avoid tokenizing again
        self.train_loader = None
        self.test_loader = None

    def train(self, train_dataset, remaining_time_budget=None, verbose=False, interval=100):
        """model training on train_dataset.

        :param train_dataset: tuple, (x_train, y_train)
            x_train: list of str, input training sentences.
            y_train: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget: float
        """
        if self.done_training:
            return

        # Running Naive classical model at the start (run_count=0)
        if self.run_count == 0:
            x_train, y_train = train_dataset

            if self.split_ratio < 1:
                # Stratified split to create representative validation set
                x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                                      test_size=1-self.split_ratio,
                                                                      random_state=1,
                                                                      stratify=ohe2cat(y_train))
                valid_dataset = (x_valid, y_valid)
                train_dataset = (x_train, y_train)

                self.preprocessed_valid_dataset = self.preprocess.preprocess_text(valid_dataset,
                                                                                  cutoff=self.str_cutoff)

            self.preprocessed_train_dataset = self.preprocess.preprocess_text(train_dataset,
                                                                              cutoff=self.str_cutoff)
            print('--' * 60)
            print('meta -> ', self.preprocess.metadata)
            print('--' * 60)
            # NaiveModel expects metadata to have 'imbalance' and must be after
            self.naive = NaiveModel(self.metadata, self.features, self.classifier)
            self.naive.train(self.preprocessed_train_dataset)

            # evaluate on validation data if available
            if self.split_ratio < 1:
                naive_valid = self.naive.test(self.preprocessed_valid_dataset[0])
                self.best_valid_score = self.score_fn(self.preprocessed_valid_dataset[1], naive_valid)
                print('Score = ', self.best_valid_score)
            self.run_count += 1
            return

        # create dataloader
        if self.train_loader is None:
            load_time = time.time()
            # preprocess
            if self.preprocessed_train_dataset is None:
                self.preprocessed_train_dataset = self.preprocess.preprocess_text(train_dataset,
                                                                                  cutoff=self.str_cutoff)
            x_train, y_train = self.preprocessed_train_dataset
            # loader
            self.train_loader = TextLoader(data=x_train, label=y_train,
                                           metadata=self.metadata,
                                           tokenizer=self.tokenizer,
                                           batch_size=self.batch_size,
                                           shuffle=True, sort=True,
                                           device=self.device, workers=self.workers)
            if self.split_ratio < 1:
                # load validation set
                if self.preprocessed_valid_dataset is None:
                    self.preprocessed_valid_dataset = self.preprocess.preprocess_text(valid_dataset,
                                                                                      cutoff=self.str_cutoff)
                x_valid, y_valid = self.preprocessed_valid_dataset
                # loader
                self.valid_loader = TextLoader(data=x_valid, label=y_valid,
                                               metadata=self.metadata,
                                               tokenizer=self.tokenizer,
                                               batch_size=self.batch_size,
                                               shuffle=True, sort=True,
                                               device=self.device, workers=self.workers)

            print('Data loading time: ', time.time() - load_time)

        # tracker to check if validation score doesn't improve against self.stop_count
        not_improved_count = 0
        # train model until it performs better than the current best valid score
        epoch_time = time.time()
        while not self.done_training:
            print('---', self.epochs + 1, '---')
            train_time = time.time()
            self._train(self.train_loader, verbose, interval)
            print('Train time:', time.time() - train_time)

            # update stats
            self.epochs += 1
            if self.epochs > self.train_epochs:
                # Signals end of program (training)
                self.done_training = True

            # validation test
            if self.split_ratio < 1:
                valid_time = time.time()
                y_valid, score = self._evaluate(self.valid_loader, verbose, interval)
                print('Valid time:', time.time() - valid_time)
                print('Score = ', score)

                if score > self.best_valid_score:
                    self.best_valid_score = score
                    # signal from the validation set score to fetch new test evaluations
                    self.update_test = True
                    break
                else:
                    self.update_test = False
                    not_improved_count += 1
                    if not_improved_count > self.stop_count:
                        # Signals end of program (training)
                        self.done_training = True
                        return
            else:
                break

        epoch_runtime = time.time() - epoch_time
        print('Total train step time: ', epoch_runtime)
        self.run_count += 1

        if remaining_time_budget is not None and remaining_time_budget < epoch_runtime:
            # Not enough time left for one more epoch
            # Signals end of program (training)
            self.done_training = True
            return

    def _train(self, loader, verbose=False, interval=100):
        ''' trains model on the given data loader '''

        self.model.train()
        loss_tracker = []
        for b, (x_batch, y_batch, mask) in enumerate(loader):
            if np.random.uniform() < self.train_batches:
                batch_time = time.time()
                self.optimizer.zero_grad()
                # forward
                preds = self.model(x_batch, mask=mask)
                # backward
                y_batch = torch.argmax(y_batch, dim=1)
                if self.metadata['class_num'] == 2:
                    loss = self.criterion(preds.view(-1), y_batch.float())  # for BCE
                else:
                    loss = self.criterion(preds, y_batch)  # for CE
                loss.backward()
                self.optimizer.step()

                if verbose:
                    if (b + 1) % interval == 0:
                        print('Epoch %d - Step [%d/%d] - loss %.6f  (dur: %.3f)' % (
                            self.epochs + 1, b + 1, len(loader), loss.item(),
                            time.time() - batch_time))

                loss_tracker.append(loss.item())

        # update batch selection probability
        if self.train_batches < 1.0:
            self.train_batches = min(1.0, self.train_batches * self.batch_alpha)

        print(self.epochs + 1, 'Avg. loss = ', np.mean(loss_tracker))

    def _evaluate(self, loader, verbose=False, interval=100, score_fn=autodl_auc):
        ''' evaluates model on the given data loader '''

        y_pred = []
        y_true = []

        self.model.eval()
        for b, (x_batch, y_batch, mask) in enumerate(loader):
            batch_time = time.time()
            # forward
            preds = self.model(x_batch, mask=mask)

            if self.metadata['class_num'] == 2:
                preds = torch.round(torch.sigmoid(preds)).view(-1).long()
            else:
                preds = torch.argmax(preds, dim=1)
            # get labels
            preds = preds.detach().cpu().numpy()
            labels = np.zeros((preds.shape[0], self.metadata['class_num']))
            labels[np.arange(preds.shape[0]), preds] = 1.0
            y_pred.append(labels)

            if y_batch is not None:
                y_true.append(y_batch.cpu().numpy())

            if verbose:
                if (b + 1) % interval == 0:
                    print('Step [%d/%d] - (dur: %.3f)' % (
                        b + 1, len(loader), time.time() - batch_time))

        y_pred = np.vstack(y_pred).astype(np.uint8)

        # score
        score = -0.99999
        if y_true:
            y_true = np.vstack(y_true).astype(np.uint8)
            score = score_fn(y_true, y_pred)
        return y_pred, score

    def test(self, x_test, remaining_time_budget=None, verbose=False, interval=100):
        """
        :param x_test: list of str, input test sentences.
        :param remaining_time_budget: float
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                 here `sample_count` is the number of examples in this dataset as test
                 set and `class_num` is the same as the class_num in metadata. The
                 values should be binary or in the interval [0,1].
        """
        if remaining_time_budget is not None and self.test_runtime is not None and remaining_time_budget < self.test_runtime:
            return self.latest_test_preds

        # TODO smarter switch
        if self.run_count < self.naive_limit:  # return naive preds for first few runs
            if self.preprocessed_test_dataset is None:
                self.preprocessed_test_dataset, _ = self.preprocess.preprocess_text((x_test, None), cutoff=90)

            if self.naive_preds is None:
                self.naive_preds = self.naive.test(self.preprocessed_test_dataset)

            self.latest_test_preds = self.naive_preds
            return self.naive_preds

        # Return previously found predictions if validation score (self.best_valid_score) hasn't improved
        if self.update_test is False:
            return self.latest_test_preds

        # create dataloader
        if self.test_loader is None:
            load_time = time.time()
            # preprocess
            if self.preprocessed_test_dataset is None:
                self.preprocessed_test_dataset, _ = self.preprocess.preprocess_text((x_test, None), cutoff=90)

            # loader
            self.test_loader = TextLoader(data=self.preprocessed_test_dataset, label=None,
                                          metadata=self.metadata,
                                          tokenizer=self.tokenizer,
                                          batch_size=self.batch_size,
                                          shuffle=False, sort=True,
                                          device=self.device, workers=self.workers)
            print('Data loading time: ', time.time() - load_time)

        epoch_time = time.time()
        y_test, _ = self._evaluate(self.test_loader, verbose, interval)
        self.test_runtime = time.time() - epoch_time
        print('Test time: ', self.test_runtime)

        # get original order back if sorted
        if self.test_loader.sort:
            orig_order = self.test_loader.indices.argsort()
            y_test = y_test[orig_order]

        self.latest_test_preds = y_test
        return y_test
