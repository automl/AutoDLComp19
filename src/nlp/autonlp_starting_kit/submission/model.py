# -*- coding: utf-8 -*-

import pytorch_transformers as pytrf
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

import re
import six
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn import metrics


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

MAX_STR_LEN = 2500
MAX_TOK_LEN = 512

BERT_PRETRAINED = {
    'EN': {'layers': 2, 'heads': 3, 'vocab': 30522, 'file': 'bert_english.model', 'name': 'bert-base-uncased'},
    'ZH': {'layers': 2, 'heads': 3, 'vocab': 21128, 'file': 'bert_chinese.model', 'name': 'bert-base-chinese'}
}


# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)


class AbstractTokenizer():
    def __init__(self):
        return

    def tokenize(self, line):
        raise NotImplementedError('No tokenize implemented!')


# class BertMultiTokenizer(AbstractTokenizer):
#     def __init__(self, name, max_len=None, max_tokens=None):
#         super().__init__()
#         self.tokenizer = pytrf.BertTokenizer.from_pretrained(name)
#         self.max_len = max_len
#         self.max_tokens = max_tokens

#     def tokenize(self, line):
#         # restrict sentence length if too long (saves read time)
#         line = line[:self.max_len]
#         line = '[CLS] '+line+' [SEP]'  # tokens required for bert
#         line = self.tokenizer.tokenize(line)
#         line = line[:self.max_tokens]
#         line = self.tokenizer.convert_tokens_to_ids(line)
#         return line


class BertTokenizer():
    def __init__(self, language='EN', pretrained_path='./'):
        super().__init__()

        name = BERT_PRETRAINED[language]['name'] + '-vocab.txt'
        # self.tokenizer = pytrf.BertTokenizer.from_pretrained(name)
        self.tokenizer = pytrf.BertTokenizer(vocab_file=os.path.join(pretrained_path, name),
                                             do_lower_case=True)

    # TODO revisit threading
    def _multithreading(self, func, args, workers):
        begin_time = time.time()
        # print("Threading with {} workers".format(workers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            res = executor.map(func, args)
        return list(res)

    def _multiprocessing(self, func, args, workers):
        begin_time = time.time()
        # print("Threading with {} workers".format(workers))
        p = mp.Pool(workers)
        res = p.map(func, args)
        p.close()
        p.join()
        return res

    def tokenize_text(self, text, max_str_len, max_tok_len=512):
        max_tok_len = max_tok_len - 3  # to account for [CLS] and [SEP]
        text = text[:max_str_len] if np.random.uniform() > 0.5 else text[-max_str_len:]
        text = self.tokenizer.tokenize(text)
        text = text[:max_tok_len] if np.random.uniform() > 0.5 else text[-max_tok_len:]
        # text = text[:max_tok_len]
        text.insert(0, self.tokenizer.vocab['[CLS]'])
        text.insert(len(text), self.tokenizer.vocab['[SEP]'])
        # print(max_tok_len, len(text))
        return self._encode_tokens(text)

    def _encode_tokens(self, text):
        return self.tokenizer.convert_tokens_to_ids(text)

    def tokenize(self, data, max_str_len, max_tok_len=512, workers=4):
        token_fn = partial(self.tokenize_text, max_str_len=max_str_len, max_tok_len=max_tok_len)
        if workers > 1:
            res = self._multiprocessing(token_fn, data, workers)
        else:
            res = [token_fn(d) for d in data]
        return res


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
        # text = [np.pad(b, (max_batch_len - len(b), 0), mode='constant') for b in text]
        text = [[0] * (max_batch_len - len(b)) + b for b in text]
        # convert to tensor
        text = torch.tensor(text).to(self.device).long()
        if self.label is not None:
            label = self.label[indices]
            label = torch.tensor(label).to(self.device).long()
            return text, label
        return text, None

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


class NLPBertClassifier(nn.Module):
    def __init__(self, metadata, pretrained=None, vocab_size=None, encoder_layers=3, attn_heads=2,
                 classifier_layers=1, classifier_units=768):
        """
        metadata: dict
            metadata from AutoNLP dataset
        pretrained: str
            folder to load a pretrained model from for the given config
        vocab_size: int or None
            vocab size for embedding layer in BERT model
            can be ignored if pretrained model is provided, else, it has to be given
        encoder_layers: int
            number of hidden encoder layers in BERT model
        attn_heads: int
            number of attention heads in BERT model
        classifier_layers: int
            number of linear layers in classifier
        classifier_units: int
            number of neurons in intermediate classifier layers
            only used when classifier_layers > 1
        """
        super().__init__()
        self.language = metadata['language']
        self.out_dim = metadata['class_num'] if metadata['class_num'] > 2 else 1

        self.layers = encoder_layers if pretrained is None else BERT_PRETRAINED[self.language]['layers']
        self.heads = attn_heads if pretrained is None else BERT_PRETRAINED[self.language]['heads']
        self.vocab_size = vocab_size if pretrained is None else BERT_PRETRAINED[self.language]['vocab']

        # load BERT
        if pretrained is not None:
            self._load_pretrained(pretrained)
        else:
            self.config = pytrf.BertConfig(self.vocab_size, num_hidden_layers=self.layers,
                                           num_attention_heads=self.heads)
            self.bert = pytrf.BertModel(self.config)

        # classifier layers
        if classifier_layers == 1:
            classifier = [nn.Linear(self.config.hidden_size, self.out_dim)]
        else:
            classifier = []
            for i in range(classifier_layers - 1):
                if i == 0:
                    fc = nn.Linear(self.config.hidden_size, classifier_units)
                else:
                    fc = nn.Linear(classifier_units, classifier_units)
                classifier.append(fc)
                classifier.append(nn.ReLU())

            classifier.append(nn.Linear(classifier_units, self.out_dim))

        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(*classifier)
        self.classifier.apply(self._init_weights)

    def forward(self, x):
        """
        input dim = (batch_size, sequence_length)
        output dim = (batch_size, num_classes)
        """
        x, _ = self.bert(x)
        # average over sequence length
        x = torch.mean(x, dim=1)
        x = self.relu(x)
        x = self.classifier(x)
        return x

    def _load_pretrained(self, pretrained_path='./'):
        """
        Loads from the pretrained model stored in given folder
        path: folder path where model is stored
        """
        self.config = pytrf.BertConfig(self.vocab_size, num_hidden_layers=self.layers)
        self.bert = pytrf.BertModel(self.config)

        # prune attention heads
        prune_heads = {}
        for i in range(self.layers):
            prune_heads[i] = np.arange(self.heads, 12).reshape(-1, 1).tolist()
        self.bert.prune_heads(prune_heads)

        # load pretrained model
        self.bert.half()
        self.bert.load_state_dict(torch.load(os.path.join(pretrained_path, BERT_PRETRAINED[self.language]['file'])))
        self.bert.float()

        # disable gradient for bert embedding layer
        for name, param in self.bert.named_parameters():
            if name.startswith('embeddings'):
                param.requires_grad = False

    def save(self, file_path='./model.pkl'):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    def count_parameters(self):
        tot_sum = sum(p.numel() for p in self.parameters())
        return tot_sum

    def trainable_parameters(self):
        tot_sum = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return tot_sum


class NaiveModel():
    def __init__(self, metadata, features):
        self.metadata = metadata
        self.classes = metadata['class_num']
        self.features = features
        self.hv = HashingVectorizer(n_features=features)
        # Run logistic (multinomial) regression for all inputs except when
        # 1) class imbalance is more than 0.2 and
        # 2) number of training samples are more than 80000
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


class Model(object):
    """
    model of BERT baseline without pretraining
    """

    def __init__(self, metadata, train_output_path="./", test_input_path="./"):
        """ Initialization for model
        :param metadata: a dict
        """
        self.done_training = False
        self.metadata = metadata
        self.train_output_path = train_output_path
        self.test_input_path = test_input_path

        self.classes = metadata['class_num']

        self.preprocessed_train_dataset = None
        self.preprocessed_test_dataset = None

        # Run tracker
        self.run_count = 0

        # Naive model parameters
        self.features = 2000

        # Parameters
        # TODO smarter params
        self.train_batches = np.inf
        self.train_epochs = np.inf
        self.naive_limit = 5
        self.batch_size = 64
        # self.pretrained_path = '/content/'
        # self.pretrained_path = ''
        # self.pretrained_path = os.path.join(*os.path.abspath(__file__).split('/')[:-1])
        self.pretrained_path = os.path.join(os.path.dirname(__file__), 'pretrained/')
        self.classifier_layers = 2
        self.classifier_units = 256
        self.learning_rate = 0.001
        self.warmum_steps = 10
        self.t_total = 500

        # create tokenizer
        #         s
        # elf.tokenizer = BertMultiTokenizer(BERT_PRETRAINED[self.metadata['language']]['name'],
        #                                             max_len=3000, max_tokens=512)
        self.tokenizer = BertTokenizer(self.metadata['language'], self.pretrained_path)

        # initialize model, optimizer
        self.model = NLPBertClassifier(metadata, pretrained=self.pretrained_path,
                                       classifier_layers=self.classifier_layers,
                                       classifier_units=self.classifier_units)
        self.model.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        #         self.optimizer = pytrf.WarmupCosineSchedule(
        #             optim.Adam(self.model.parameters(), lr=self.learning_rate),
        #             warmup_steps=self.warmum_steps, t_total=self.t_total)

        self.criterion = nn.BCEWithLogitsLoss() if self.metadata['class_num'] == 2 \
            else nn.CrossEntropyLoss()

        # to store train dataset to avoid tokenizing again
        self.train_loader = None
        self.test_loader = None

        # initialize stats
        self.epochs = 0

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")

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
        REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')

        if data_y is None:
            data_y = [0 for i in range(len(data_x))]

        ret = []
        class_freq = [0 for i in range(self.classes)]
        len_list = []
        for i, line in enumerate(data_x):
            line = self.convert_to_unicode(line)
            # line = line.decode('utf-8', 'ignore')
            line = line.lower()  # lowercase text
            line = REPLACE_BY_SPACE_RE.sub(' ', line)
            line = BAD_SYMBOLS_RE.sub('', line)
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
        REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')

        if data_y is None:
            data_y = [0 for i in range(len(data_x))]

        ret = []
        class_freq = [0 for i in range(self.classes)]
        len_list = []
        for i, line in enumerate(data_x):
            line = self.convert_to_unicode(line)
            # line = line.decode('utf-8', 'ignore')
            line = REPLACE_BY_SPACE_RE.sub(' ', line)
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

    def train(self, train_dataset, remaining_time_budget=None, verbose=False):
        """model training on train_dataset.

        :param train_dataset: tuple, (x_train, y_train)
            x_train: list of str, input training sentences.
            y_train: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
        """
        if self.done_training:
            return

        # Running Naive classical model at the start (train_run_count=0)
        if self.run_count == 0:
            x_train, y_train = train_dataset
            self.preprocessed_train_dataset = self.preprocess_text(train_dataset, cutoff=90)
            # NaiveModel expects metadata to have 'imbalance' and must be after
            self.naive = NaiveModel(self.metadata, self.features)
            self.naive.train(self.preprocessed_train_dataset)
            self.run_count += 1
            return

        # create dataloader
        if self.train_loader is None:
            load_time = time.time()
            # preprocess
            if self.preprocessed_train_dataset is not None:
                self.preprocessed_train_dataset = self.preprocess_text(train_dataset, cutoff=90)
            x_train, y_train = self.preprocessed_train_dataset
            # loader
            self.train_loader = TextLoader(data=x_train, label=y_train,
                                           metadata=self.metadata,
                                           tokenizer=self.tokenizer,
                                           batch_size=self.batch_size,
                                           shuffle=True, sort=True)
            print('Data loading time: ', time.time() - load_time)

        # train model
        epoch_time = time.time()
        self._train(self.train_loader, verbose)
        print('Train time: ', time.time() - epoch_time)

        # update stats
        self.epochs += 1
        if self.epochs > self.train_epochs:
            self.done_training = True

        self.run_count += 1

    def _train(self, loader, verbose=False, interval=50):

        self.model.train()
        loss_tracker = []
        for b, (x_batch, y_batch) in enumerate(loader):
            batch_time = time.time()
            self.optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # forward
            preds = self.model(x_batch)
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

            if b >= self.train_batches:
                print('Interrupting training beacuse batch limit reached...')
                break

        print(self.epochs, 'Avg. loss = ', np.mean(loss_tracker))

    def test(self, x_test, remaining_time_budget=None, verbose=False, interval=1):
        """
        :param x_test: list of str, input test sentences.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                 here `sample_count` is the number of examples in this dataset as test
                 set and `class_num` is the same as the class_num in metadata. The
                 values should be binary or in the interval [0,1].
        """
        # TODO smarter switch
        if self.run_count < self.naive_limit:  # 1 classical + 3 transformer epochs
            self.preprocessed_test_dataset, _ = self.preprocess_text((x_test, None), cutoff=90)
            preds = self.naive.test(self.preprocessed_test_dataset)
            return preds

        # create dataloader
        if self.test_loader is None:
            load_time = time.time()
            # preprocess
            if self.preprocessed_test_dataset is not None:
                self.preprocessed_test_dataset = self.preprocess_text((x_test, None), cutoff=90)
            # loader
            self.test_loader = TextLoader(data=self.preprocessed_test_dataset[0], label=None,
                                          metadata=self.metadata,
                                          tokenizer=self.tokenizer,
                                          batch_size=self.batch_size,
                                          shuffle=False, sort=True)
            print('Data loading time: ', time.time() - load_time)

        y_test = []

        # get predictions
        self.model.eval()
        epoch_time = time.time()
        for b, (x_batch, _) in enumerate(self.test_loader):
            batch_time = time.time()
            x_batch = x_batch.to(device)
            # forward
            preds = self.model(x_batch)

            if self.metadata['class_num'] == 2:
                preds = torch.round(torch.sigmoid(preds)).view(-1).long()
            else:
                preds = torch.argmax(preds, dim=1)
            # get labels
            preds = preds.detach().cpu().numpy()
            labels = np.zeros((preds.shape[0], self.metadata['class_num']))
            labels[np.arange(preds.shape[0]), preds] = 1.0
            y_test.append(labels)

            if verbose:
                if (b + 1) % interval == 0:
                    print('Step [%d/%d] - (dur: %.3f)' % (
                        b + 1, len(self.test_loader), time.time() - batch_time))

        print('Test time: ', time.time() - epoch_time)

        y_test = np.vstack(y_test).astype(np.uint8)

        # get original order back if sorted
        if self.test_loader.sort:
            orig_order = self.test_loader.indices.argsort()
            y_test = y_test[orig_order]

        return y_test
