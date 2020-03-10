#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/9/24 15:12
# @Author:  Mecthew

import librosa
import numpy as np
import tensorflow as tf
from data_process import extract_mfcc_parallel, get_max_length, ohe2cat, pad_seq
from models.my_classifier import Classifier
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.layers import (
    Activation, BatchNormalization, Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPool1D,
    MaxPooling2D
)
from tensorflow.python.keras.models import Sequential
from tools import log, timeit


class CnnModel2D(Classifier):
    def __init__(self):
        # clear_session()
        self.max_length = None

        self._model = None
        self.is_init = False

    def init_model(self, input_shape, num_classes, model_config, max_layer_num=5, **kwargs):
        # FIXME: keras sequential model is better than keras functional api,
        # why???
        model = Sequential()
        min_size = min(input_shape[:2])
        for i in range(max_layer_num):
            if i == 0:
                model.add(Conv2D(64, 3, input_shape=input_shape, padding='same'))
            else:
                model.add(Conv2D(64, 3, padding='same'))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            min_size //= 2
            if min_size < 2:
                break

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dropout(rate=0.5))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        self.model_config = model_config

        # optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6)

        # Thomas' comment: copied parameters from default constructor
        optimizer = tf.keras.optimizers.Adam(
            lr = self.model_config["optimizer"]["lr_cnn"],
            beta_1 = 1-self.model_config["optimizer"]["beta_1"],
            beta_2 = 1-self.model_config["optimizer"]["beta_2"],
            epsilon = self.model_config["optimizer"]["epsilon"],
            decay = self.model_config["optimizer"]["decay"],
            amsgrad = self.model_config["optimizer"]["amsgrad"]
        )
        # optimizer = optimizers.SGD(lr=1e-3, decay=2e-4, momentum=0.9, clipvalue=5)
        model.compile(
            loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']
        )
        model.summary()
        self.is_init = True
        self._model = model

    def preprocess_data(self, x):
        if self.model_config["common"]["is_cut_audio"]:
            x = [sample[0:self.model_config["common"]["max_audio_duration"] * self.model_config["common"]["audio_sample_rate"]] for sample in x]
        # extract mfcc
        x = extract_mfcc_parallel(x,
                                  sr=self.model_config["common"]["sr"],
                                  fft_duration=self.model_config["common"]["fft_duration"],
                                  hop_duration=self.model_config["common"]["hop_duration"],
                                  n_mfcc=self.model_config["common"]["num_mfcc"])
        if self.max_length is None:
            self.max_length = get_max_length(x)
        x = pad_seq(x, self.max_length)

        # if self.scaler is None:
        #     self.scaler = []
        #     for i in range(x.shape[2]):
        #         self.scaler.append(StandardScaler().fit(x[:, :, i]))
        # for i in range(x.shape[2]):
        #     x[:, :, i] = self.scaler[i].transform(x[:, :, i])

        # feature scale
        # if self.mean is None or self.std is None:
        #     self.mean = np.mean(x)
        #     self.std = np.std(x)
        #     x = (x - self.mean) / self.std

        # s0, s1, s2 = x.shape[0], x.shape[1], x.shape[2]
        # x = x.reshape(s0 * s1, s2)
        # if not self.scaler:
        #     self.scaler = MinMaxScaler().fit(x)
        # x = self.scaler.transform(x)
        # x = x.reshape(s0, s1, s2)

        # 4 dimension?
        # (120, 437, 24) to (120, 437, 24, 1)
        # 120 is the number of instance
        # 437 is the max length
        # 24 frame in mfcc
        # log(f"max {np.max(x)} min {np.min(x)} mean {np.mean(x)}")

        x = x[:, :, :, np.newaxis]
        return x

    def fit(self, train_x, train_y, validation_data_fit, train_loop_num, **kwargs):
        val_x, val_y = validation_data_fit

        # if train_loop_num == 1:
        #     patience = 2
        #     epochs = 8
        # elif train_loop_num == 2:
        #     patience = 3
        #     epochs = 10
        # elif train_loop_num < 10:
        #     patience = 4
        #     epochs = 16
        # elif train_loop_num < 15:
        #     patience = 4
        #     epochs = 24
        # else:
        #     patience = 8
        #     epochs = 32

        epochs = 3
        patience = 2

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)]

        self._model.fit(
            train_x,
            ohe2cat(train_y),
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(val_x, ohe2cat(val_y)),
            verbose=1,  # Logs once per epoch.
            batch_size=32,
            shuffle=True
        )

    def predict(self, x_test, batch_size=32):
        return self._model.predict(x_test, batch_size=batch_size)
