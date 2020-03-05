"""MIT License

Copyright (c) 2019 Lenovo Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
# -*- coding: utf-8 -*-

import argparse
import getopt
import gzip
import logging
import os
import re
import sys
import time
import pickle
from functools import reduce

import keras
import numpy as np
import pandas as pd
from data_manager import DataGenerator
from keras import backend as K
from keras.preprocessing import \
    sequence  # from tensorflow.python.keras.preprocessing import sequence
from model_manager import ModelGenerator

# fmt: off
os.system("pip install jieba_fast")  # isort:skip
import jieba_fast as jieba  # isort:skip
# fmt: on

print(keras.__version__)

# from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
# (nothing gets printed in Jupyter, only if you run it standalone)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

# Limit on the number of features. We use the top 20K features

verbosity_level = 'INFO'


def get_logger(verbosity_level, use_error_log=False):
    """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO score.py: <message>
  """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger(verbosity_level)


def tiedrank(a):
    ''' Return the ranks (with base 1) of a list resolving ties by averaging.
     This works for numpy arrays.'''
    m = len(a)
    # Sort a in ascending order (sa=sorted vals, i=indices)
    i = a.argsort()
    sa = a[i]
    # Find unique values
    uval = np.unique(a)
    # Test whether there are ties
    R = np.arange(m, dtype=float) + 1  # Ranks with base 1
    if len(uval) != m:
        # Average the ranks for the ties
        oldval = sa[0]
        k0 = 0
        for k in range(1, m):
            if sa[k] != oldval:
                R[k0:k] = sum(R[k0:k]) / (k - k0)
                k0 = k
                oldval = sa[k]
        R[k0:m] = sum(R[k0:m]) / (m - k0)
    # Invert the index
    S = np.empty(m)
    S[i] = R
    return S


def mvmean(R, axis=0):
    ''' Moving average to avoid rounding errors. A bit slow, but...
    Computes the mean along the given axis, except if this is a vector, in which case the mean is returned.
    Does NOT flatten.'''
    if len(R.shape) == 0:
        return R
    average = lambda x: reduce(
        lambda i, j: (0, (j[0] / (j[0] + 1.)) * i[1] + (1. / (j[0] + 1)) * j[1]), enumerate(x)
    )[1]
    R = np.array(R)
    if len(R.shape) == 1:
        return average(R)
    if axis == 1:
        return np.array(map(average, R))
    else:
        return np.array(map(average, R.transpose()))


# code form https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
def clean_en_text(dat, max_seq_length, ratio=0.1, is_ratio=True):

    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')

    ret = []
    for line in dat:
        # text = text.lower() # lowercase text
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        # line = BAD_SYMBOLS_RE.sub('', line)
        line = line.strip()
        line_split = line.split()

        if is_ratio:
            NUM_WORD = max(int(len(line_split) * ratio), max_seq_length)
        else:
            NUM_WORD = max_seq_length

        if len(line_split) > NUM_WORD:
            line = " ".join(line_split[0:NUM_WORD])
        ret.append(line)
    return ret


def clean_zh_text(dat, max_char_length, ratio=0.1, is_ratio=False):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')

    ret = []
    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = line.strip()

        if is_ratio:
            NUM_CHAR = max(int(len(line) * ratio), max_char_length)
        else:
            NUM_CHAR = max_char_length

        if len(line) > NUM_CHAR:
            # line = " ".join(line.split()[0:MAX_CHAR_LENGTH])
            line = line[0:NUM_CHAR]
        ret.append(line)
    return ret


def _tokenize_chinese_words(text):
    return ' '.join(jieba.cut(text, cut_all=False))


# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)


class Model(object):
    """
        model of CNN baseline without pretraining.
        see `https://aclweb.org/anthology/D14-1181` for more information.
    """

    def __init__(self, metadata, model_config, train_output_path="./", test_input_path="./"):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 10,
             "language": ZH,
             "num_train_instances": 10000,
             "num_test_instances": 1000,
             "time_budget": 300}
        """
        self.done_training = False
        self.metadata = metadata
        self.model_config = model_config["autonlp"]
        print('-----------')
        print(self.model_config)
        print('-----------')

        self.train_output_path = train_output_path
        self.test_input_path = test_input_path
        self.model = None
        self.call_num = 0
        self.load_pretrain_emb = True
        self.batch_size = self.model_config["model"]["init_batch_size"]
        self.total_call_num = self.model_config["model"]["total_call_num"]
        self.valid_cost_list = []
        self.auc = 0
        self.svm = True
        self.svm_model = None
        self.svm_token = None
        self.tokenizer = None
        self.model_weights_list = []
        # 0: char based   1: word based   2: doc based
        self.feature_mode = 1

        # "text_cnn" "lstm" "sep_cnn_model"
        self.model_mode = 'text_cnn'
        self.fasttext_embeddings_index = None

        # 0: binary_crossentropy
        # 1: categorical_crossentropy
        # 2: sparse_categorical_crossentropy
        self.metric = 1

        self.num_features = self.model_config["common"]["max_vocab_size"]
        # load pretrian embeding

        if self.load_pretrain_emb:
            self._load_emb()


    def train(self, train_dataset, remaining_time_budget=None):
        """model training on train_dataset.It can be seen as metecontroller

        :param train_dataset: tuple, (x_train, y_train)
            x_train: list of str, input training sentences.
            y_train: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
:        """
        if self.done_training:
            return

        if self.call_num == 0:
            self.data_generator = DataGenerator(train_dataset, self.metadata, self.model_config)

        x_train, y_train = self.data_generator.sample_dataset_from_metadataset()
        x_train, feature_mode = self.data_generator.dataset_preporocess(x_train)

        if self.call_num == 0:
            #self.data_generator.dataset_postprocess(x_train,y_train,'svm')
            self.model_manager = ModelGenerator(
                self.data_generator.feature_mode,
                self.model_config,
                load_pretrain_emb=self.load_pretrain_emb,
                fasttext_embeddings_index=self.fasttext_embeddings_index
            )

        self.model_name = self.model_manager.model_pre_select(self.call_num)
        #self.svm_token = self.data_generator.svm_token
        self.data_generator.dataset_postprocess(x_train, y_train, self.model_name)
        if self.call_num <= 1:
            self.model = self.model_manager.build_model(
                self.model_name, self.data_generator.data_feature
            )

        #self.data_generator.dataset_postprocess()

        if self.model_name == 'svm':
            self.model.fit(self.data_generator.x_train, ohe2cat(self.data_generator.y_train))
            self.svm_token = self.data_generator.svm_token
            valid_auc = self._valid_auc(
                self.data_generator.valid_x, self.data_generator.valid_y, svm=True
            )
            self.valid_auc_svm = valid_auc
            print("valid_auc_svm", self.valid_auc_svm)
            # self.svm = False
        else:
            callbacks = None
            history = self.model.fit(
                self.data_generator.x_train,
                self.data_generator.y_train,
                epochs=self.model_config["model"]["num_epoch"],
                callbacks=callbacks,
                validation_split=self.model_config["model"]["valid_ratio"],
                validation_data=(self.data_generator.valid_x, self.data_generator.valid_y),
                verbose=2,
                batch_size=self.batch_size,
                shuffle=True
            )

            self.feedback_simulation(history,
                                     self.model_config["model"]["increase_batch_acc"],
                                     self.model_config["model"]["early_stop_auc"])

    def _get_valid_columns(self, solution):
        """Get a list of column indices for which the column has more than one class.
        This is necessary when computing BAC or AUC which involves true positive and
        true negative in the denominator. When some class is missing, these scores
        don't make sense (or you have to add an epsilon to remedy the situation).

        Args:
          solution: array, a matrix of binary entries, of shape
            (num_examples, num_features)
        Returns:
          valid_columns: a list of indices for which the column has more than one
            class.
        """
        num_examples = solution.shape[0]
        col_sum = np.sum(solution, axis=0)
        valid_columns = np.where(1 - np.isclose(col_sum, 0) - np.isclose(col_sum, num_examples))[0]
        return valid_columns

    def _autodl_auc(self, solution, prediction, valid_columns_only=True):
        """Compute normarlized Area under ROC curve (AUC).
        Return Gini index = 2*AUC-1 for  binary classification problems.
        Should work for a vector of binary 0/1 (or -1/1)"solution" and any discriminant values
        for the predictions. If solution and prediction are not vectors, the AUC
        of the columns of the matrices are computed and averaged (with no weight).
        The same for all classification problems (in fact it treats well only the
        binary and multilabel classification problems). When `valid_columns` is not
        `None`, only use a subset of columns for computing the score.
        """
        if valid_columns_only:
            valid_columns = self._get_valid_columns(solution)
            if len(valid_columns) < solution.shape[-1]:
                logger.warning(
                    "Some columns in solution have only one class, " +
                    "ignoring these columns for evaluation."
                )
            solution = solution[:, valid_columns].copy()
            prediction = prediction[:, valid_columns].copy()
        label_num = solution.shape[1]
        auc = np.empty(label_num)
        for k in range(label_num):
            r_ = tiedrank(prediction[:, k])
            s_ = solution[:, k]
            if sum(s_) == 0:
                print("WARNING: no positive class example in class {}".format(k + 1))
            npos = sum(s_ == 1)
            nneg = sum(s_ < 1)
            auc[k] = (sum(r_[s_ == 1]) - npos * (npos + 1) / 2) / (nneg * npos)
        return 2 * mvmean(auc) - 1

    def _valid_auc(self, x_valid, y_valid, svm=False):

        if svm:
            x_valid = self.svm_token.transform(x_valid)

            result = self.model.predict_proba(x_valid)
        else:
            result = self.model.predict(x_valid)

        return self._autodl_auc(y_valid, result)  # y_test

    def test(self, x_test, remaining_time_budget=None):
        """
        :param x_test: list of str, input test sentences.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                 here `sample_count` is the number of examples in this dataset as test
                 set and `class_num` is the same as the class_num in metadata. The
                 values should be binary or in the interval [0,1].
        """
        # model = models.load_model(self.test_input_path + 'model.h5')

        train_num, self.test_num = self.metadata['train_num'], self.metadata['test_num']
        self.class_num = self.metadata['class_num']
        print("num_samples_test:", self.test_num)
        print("num_class_test:", self.class_num)

        #if self.call_num == 0 or self.call_num == 1:
        if self.call_num == 0:
            # tokenizing Chinese words
            if self.metadata['language'] == 'ZH':
                x_test = clean_zh_text(x_test, self.model_config["common"]["max_char_length"])
                if self.data_generator.feature_mode == 1:
                    x_test = list(map(_tokenize_chinese_words, x_test))
            else:
                x_test = clean_en_text(x_test, self.model_config["common"]["max_seq_length"])
            self.x_test_clean = x_test

            x_test = self.svm_token.transform(x_test)
            result = self.model.predict_proba(x_test)
            self.svm_result = result
            self.call_num = self.call_num + 1
            return result  # y_test
        if self.call_num == 1:
            self.tokenizer = self.data_generator.tokenizer
            x_test = self.tokenizer.texts_to_sequences(self.x_test_clean)
            self.x_test = sequence.pad_sequences(
                x_test, maxlen=self.data_generator.data_feature['max_length']
            )

        if self.selcet_svm:
            result = self.svm_result
            print("load svm again!!!")
        else:
            result = self.model.predict(self.x_test, batch_size=512)

        # Cumulative training times
        self.call_num = self.call_num + 1
        if self.call_num >= self.total_call_num:
            self.done_training = True

        return result  # y_test

    def _load_emb(self):
        # loading pretrained embedding

        if self.metadata['language'] == 'ZH':
            ft_file = 'cc.zh.300.vec.gz'
            emb_file = 'ZH.pkl'
        else:
            ft_file = 'cc.en.300.vec.gz'
            emb_file = 'EN.pkl'

        emb_path = None
        for ft_dir in self.model_config["model"]["ft_dir"]:
            if os.path.isfile(os.path.join(ft_dir, emb_file)):
                emb_path = os.path.join(ft_dir, emb_file)
                break

        fasttext_embeddings_index = {}
        if emb_path is None:
            print('found no embedding file')
            f = None
            for ft_dir in self.model_config["model"]["ft_dir"]:
                ft_path = os.path.join(ft_dir, ft_file)
                if os.path.isfile(ft_path):
                    f = gzip.open(ft_path, 'rb')
                    break

            if f is None:
                raise ValueError("Embedding not found")

            for line in f.readlines():
                values = line.strip().split()
                word = values[0].decode('utf8')
                coefs = np.asarray(values[1:], dtype='float32')
                fasttext_embeddings_index[word] = coefs
        else:
            print('found embedding file')
            fasttext_embeddings_index = pickle.load(open(emb_path, "rb"))


        print('Found %s fastText word vectors.' % len(fasttext_embeddings_index))
        self.fasttext_embeddings_index = fasttext_embeddings_index

        # embedding lookup
        #EMBEDDING_DIM = self.emb_size
        #self.embedding_matrix = np.zeros((self.num_features, EMBEDDING_DIM))
        #return self.embedding_matrix

    def feedback_simulation(self, history, increase_batch_acc=None, early_stop_auc=None):
        # Model Selection and Sample num from Feedback Dynamic Regulation of Simulator
        # Dynamic sampling ,if accuracy is lower than 0.65 ,Increase sample size
        self.sample_num_per_class = self.data_generator.sample_num_per_class
        for key in ['acc', 'accuracy']:
            if key in history.history.keys():
                if history.history[key][0] < increase_batch_acc:
                    self.sample_num_per_class = min(
                        4 * self.data_generator.sample_num_per_class,
                        self.data_generator.max_sample_num_per_class
                    )

        #TODO self.sample_num_per_class
        self.data_generator.set_sample_num_per_class(self.sample_num_per_class)

        # Early stop and restore weight automatic
        valid_auc = self._valid_auc(self.data_generator.valid_x, self.data_generator.valid_y)
        print("valid_auc: ", valid_auc)

        # select which model is activated
        self.selcet_svm = self.valid_auc_svm > valid_auc

        early_stop_conditon2 = self.call_num >= 3 and (
            self.valid_cost_list[self.call_num - 2] - valid_auc
        ) > 0 and (
            self.valid_cost_list[self.call_num - 3] - self.valid_cost_list[self.call_num - 2]
        ) > 0
        pre_auc = self.auc
        self.auc = valid_auc
        self.valid_cost_list.append(valid_auc)
        early_stop_conditon1 = self.auc < pre_auc and self.auc > early_stop_auc
        if early_stop_conditon1 or early_stop_conditon2:
            print('EARLY OUT')
            self.done_training = True
            if early_stop_conditon2:
                self.model.set_weights(self.model_weights_list[self.call_num - 3])
                print("load weight...")
            if self.call_num >= 1 and early_stop_conditon1:
                self.model.set_weights(self.model_weights_list[self.call_num - 2])
                print("load weight...")

        #print(str(type(self.x_train)) + " " + str(y_train.shape))

        model_weights = self.model.get_weights()
        self.model_weights_list.append(model_weights)
