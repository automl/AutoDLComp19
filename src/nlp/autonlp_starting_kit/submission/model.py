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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split

from scoring import autodl_auc
from bert import BertTokenizer, NLPBertClassifier, BERT_PRETRAINED
from xlnet import XLNetTokenizer, NLPXLNetClassifier, XLNET_PRETRAINED
from adabound import AdaBound, AdaBoundW
from radam import RAdam
from text_augmentation import Augmentation

SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MAX_STR_LEN = 2500
MAX_TOK_LEN = 512
MAX_VALID_SIZE = 10000

try:
    from apex import amp
    apex_exists = True
except Exception:
    apex_exists = False


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def ohe2cat(label):
    """ onhot encode to category """
    return np.argmax(label, axis=1)


def bucket_shuffle(arr, bucket_size, pad_key=-1):
    """ shuffle 1-D array in buckets """

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
    def __init__(self, data, label, metadata, tokenizer, augmentation=False, batch_size=1,
                 shuffle=False, sort=False, device=torch.device('cpu'), workers=4):
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

        # data augmentation
        if augmentation:
            print("Augmenting data...")
            augment_time = time.time()
            aug = Augmentation(imbalance=metadata['imbalance'], threshold=metadata['aug_threshold'])
            data, label = aug.augment_data(data, label)
            print("Augmentation time:", time.time()-augment_time)

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
            self.indices = bucket_shuffle(self.indices, bucket_size=int(self.batch_size*1.1))
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
    def __init__(self, metadata, features, config):
        self.metadata = metadata
        self.classes = metadata['class_num']
        self.train_samples = config['train_samples'] if config['train_samples'] else 100000
        self.features = features
        self.hv = HashingVectorizer(n_features=features)

        # build model based on config
        if config['classifier'] == 'auto':
            # # Run logistic (multinomial) regression for all inputs except when
            # # 1) class imbalance is more than 0.2 and
            # # 2) number of training samples are more than 80000
            if max(self.metadata['imbalance']) > 0.2 or self.metadata['train_num'] < 50000:
                self.clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
            else:
                self.clf = AdaBoostClassifier(n_estimators=50, learning_rate=1)
        else:
            if config['classifier'] == 'lr':
                self.clf = LogisticRegression(solver=config['lr_opt'], multi_class=config['lr_multi'])
            elif config['classifier'] == 'ada':
                self.clf = AdaBoostClassifier(n_estimators=config['ada_estimators'], learning_rate=config['ada_rate'])
            else:
                self.clf = SVC(kernel=config['svc_kernel'], gamma='auto', decision_function_shape='ovr')

    def _transform(self, data):
        return self.hv.transform(data)

    def train(self, data):
        data_x, data_y = data
        # sample only 'train_samples' data for training
        if self.metadata['train_num'] > self.train_samples:
            remove_ratio = 1 - self.train_samples/self.metadata['train_num']
            data_x, _, data_y, _ = train_test_split(data_x, data_y, test_size=remove_ratio,
                                                    random_state=1, stratify=ohe2cat(data_y))
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
        return data_x, data_y

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
            data_y = np.zeros(len(data_x))

        ret = []
        class_freq = np.zeros(len(self.classes))
        len_list = []
        for i, line in enumerate(data_x):
            line = convert_to_unicode(line)
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

    def _clean_zh_text(self, data_x, data_y, cutoff=90):
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
            data_y = np.zeros(len(data_x))

        ret = []
        class_freq = np.zeros(len(self.classes))
        len_list = []
        for i, line in enumerate(data_x):
            line = convert_to_unicode(line)
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
        self.done_training = False
        self.metadata = metadata
        self.train_output_path = train_output_path
        self.test_input_path = test_input_path

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.device = torch.device('cpu')

        # self.pretrained_path = '/content/'
        self.pretrained_path = os.path.join(os.path.dirname(__file__), 'pretrained/')
        self.score_fn = autodl_auc

        self.preprocess = Preprocess(metadata)

        self.preprocessed_train_dataset = None
        self.preprocessed_test_dataset = None
        self.naive_preds = None
        self.update_test = False
        self.best_valid_score = -1

        # Run tracker
        self.run_count = 0
        self.epochs = 0

        ## Parameters
        # TODO smarter params
        self.train_batches = 1.0  # prob of executing a batch during an epoch
        self.batch_alpha = 1.2  # multiplier to update batch training probability
        self.train_epochs = np.inf  # num of epochs to train before done_training=True
        self.naive_limit = 1  # num of train runs before testing using network
        self.test_runtime = None  # time taken for inference of test_dataset
        self.initialized = False
        self.workers = 1 if self.metadata['language'] == 'ZH' else 3  # since multiprocessing is poor for chinese

        # to split train into train & validation
        self.split_ratio = split_ratio
        if self.split_ratio:
            # limit validation set size to limit
            self.split_ratio = min(1 - self.split_ratio, MAX_VALID_SIZE / self.metadata['train_num'])
        print('Validation split -', self.split_ratio)

        if config is None:
            # setting default config if not provided
            config = {'transformer': 'bert', 'layers': 2, 'finetune_wait': 3,
                      'classifier_layers': 2, 'classifier_units': 256,
                      'optimizer': 'adabound', 'learning_rate': 0.001, 'weight_decay': 0.0001,
                      'batch_size': 64, 'classifier': 'auto', 'features': 2000, 'train_samples': 100000,
                      'str_cutoff': 90, 'stop_count': 25, 'augmentation': True, 'aug_threshold': 0.1}
        self.config = config

        # True/False as str to account for categorical parameter in ConfigSpace
        if self.config['augmentation']:
            self.metadata['augmentation'] = True
            self.metadata['aug_threshold'] = self.config['aug_threshold']
        else:
            self.metadata['augmentation'] = False

        self.classifier = self.config['classifier'] if self.config['classifier'] else None

        # to store train dataset and avoid tokenizing again
        self.train_loader = None
        self.test_loader = None

    def _init_models(self):
        """ initialize model & tokenizer for training """

        # initialize model & tokenizer
        if self.config['transformer'] == 'bert':
            self.metadata_pretrain = BERT_PRETRAINED
            self.metadata_pretrain['layers'] = self.config['layers']
            self.tokenizer = BertTokenizer(self.metadata['language'], self.pretrained_path)
            self.model = NLPBertClassifier(self.metadata, bert_metadata=self.metadata_pretrain,
                                           pretrained=self.pretrained_path,
                                           classifier_layers=self.config['classifier_layers'],
                                           classifier_units=self.config['classifier_units'], finetuning=False)
        elif self.config['transformer'] == 'xlnet':
            self.metadata_pretrain = XLNET_PRETRAINED
            self.metadata_pretrain['layers'] = self.config['layers']
            self.tokenizer = XLNetTokenizer(self.metadata['language'], self.pretrained_path)
            self.model = NLPXLNetClassifier(self.metadata, xlnet_metadata=self.metadata_pretrain,
                                            pretrained=self.pretrained_path,
                                            classifier_layers=self.config['classifier_layers'],
                                            classifier_units=self.config['classifier_units'], finetuning=False)

        # initialize optimizer & loss fn
        self._init_optim()
        self.criterion = nn.BCEWithLogitsLoss() if self.metadata['class_num'] == 2 \
            else nn.CrossEntropyLoss()

        if apex_exists:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                        opt_level="O2", loss_scale="dynamic")
            print("Apexifying model and optimizer")

        self.model.to(self.device)
        self.initialized = True

    def _init_optim(self):
        """ initialize optimizer """
        if self.config['optimizer'] == 'adabound':
            self.optimizer = AdaBoundW(self.model.parameters(), lr=self.config['learning_rate'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'radam':
            self.optimizer = RAdam(self.model.parameters(), lr=self.config['learning_rate'])
        elif self.config['optimizer'] == "adamw":
            self.optimizer = pytrf.AdamW(self.model.parameters(), lr=self.config['learning_rate'],
                                         weight_decay=self.config['weight_decay'])
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

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

        torch.cuda.empty_cache()

        # Running Naive classical model at the start (run_count=0)
        if self.run_count == 0:
            x_train, y_train = train_dataset

            if self.split_ratio:
                # Stratified split to create representative validation set
                x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                                      test_size=self.split_ratio,
                                                                      random_state=1,
                                                                      stratify=ohe2cat(y_train))
                valid_dataset = (x_valid, y_valid)
                train_dataset = (x_train, y_train)

                self.preprocessed_valid_dataset = self.preprocess.preprocess_text(valid_dataset,
                                                                                  cutoff=self.config['str_cutoff'])

            self.preprocessed_train_dataset = self.preprocess.preprocess_text(train_dataset,
                                                                              cutoff=self.config['str_cutoff'])
            print('--' * 60)
            print('meta -> ', self.preprocess.metadata)
            print('--' * 60)

            # run naive model if configured
            if self.classifier:
                print('Building naive model - ', self.classifier)
                naive_time = time.time()
                # NaiveModel expects metadata to have 'imbalance' and must be after
                self.naive = NaiveModel(self.metadata, self.config['features'], self.config)
                self.naive.train(self.preprocessed_train_dataset)

                # evaluate on validation data if available
                if self.split_ratio:
                    naive_valid = self.naive.test(self.preprocessed_valid_dataset[0])
                    self.best_valid_score = self.score_fn(self.preprocessed_valid_dataset[1], naive_valid)
                    print('Score = ', self.best_valid_score)
                print('Naive model time:', time.time() - naive_time)
                self.run_count += 1
                return

        # initialize model if not initialized in init
        if not self.initialized:
            self._init_models()

        # create dataloader
        if self.train_loader is None:
            load_time = time.time()
            # preprocess
            if self.preprocessed_train_dataset is None:
                self.preprocessed_train_dataset = self.preprocess.preprocess_text(train_dataset,
                                                                                  cutoff=self.config['str_cutoff'])
            x_train, y_train = self.preprocessed_train_dataset
            # loader
            self.train_loader = TextLoader(data=x_train, label=y_train,
                                           metadata=self.metadata,
                                           tokenizer=self.tokenizer,
                                           augmentation=self.metadata['augmentation'],
                                           batch_size=self.config['batch_size'],
                                           shuffle=True, sort=True,
                                           device=self.device, workers=self.workers)
            if self.split_ratio:
                # load validation set
                if self.preprocessed_valid_dataset is None:
                    self.preprocessed_valid_dataset = self.preprocess.preprocess_text(valid_dataset,
                                                                                      cutoff=self.config['str_cutoff'])
                x_valid, y_valid = self.preprocessed_valid_dataset
                # loader
                self.valid_loader = TextLoader(data=x_valid, label=y_valid,
                                               metadata=self.metadata,
                                               tokenizer=self.tokenizer,
                                               augmentation=False,
                                               batch_size=self.config['batch_size'],
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

            if self.epochs > self.train_epochs:
                # Signals end of program (training)
                self.done_training = True

            # validation test
            if self.split_ratio:
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
                    if not_improved_count > self.config['stop_count']:
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
        """ trains model on the given data loader """

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

                if apex_exists:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # optimizer step
                self.optimizer.step()

                if verbose:
                    if (b + 1) % interval == 0:
                        print('Epoch %d - Step [%d/%d] - loss %.6f  (dur: %.3f)' % (
                            self.epochs + 1, b + 1, len(loader), loss.item(),
                            time.time() - batch_time))

                loss_tracker.append(loss.item())

        # update batch selection probability
        if self.train_batches:
            self.train_batches = min(1.0, self.train_batches * self.batch_alpha)

        # update stats
        self.epochs += 1
        if not self.model.finetuning and self.epochs >= self.config['finetune_wait']:
            print('enable finetuning transformer')
            self.model.enable_finetuning()
            self._init_optim()  # reinitialize optimizer since model params changed

        print('Avg. loss = ', np.mean(loss_tracker))

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
        if remaining_time_budget is not None and \
                self.test_runtime is not None and \
                remaining_time_budget < self.test_runtime:
            return self.latest_test_preds

        torch.cuda.empty_cache()

        if self.run_count == 1:  # return naive preds for first run
            if self.preprocessed_test_dataset is None:
                self.preprocessed_test_dataset, _ = self.preprocess.preprocess_text((x_test, None), cutoff=90)

            if self.classifier and self.naive_preds is None:
                print('Generating naive predictions')
                self.naive_preds = self.naive.test(self.preprocessed_test_dataset)
                self.latest_test_preds = self.naive_preds
                return self.naive_preds

        # Return previously found predictions if validation score (self.best_valid_score) hasn't improved
        if not self.update_test and self.latest_test_preds is not None:
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
                                          augmentation=False,
                                          batch_size=self.config['batch_size'],
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
