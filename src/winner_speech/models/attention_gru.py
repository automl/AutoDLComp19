#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/10/15 22:44
# @Author:  Mecthew

import tensorflow as tf
from data_process import extract_mfcc_parallel, get_max_length, ohe2cat, pad_seq
from models.attention import Attention
from models.my_classifier import Classifier
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import (
    Bidirectional, Concatenate, CuDNNLSTM, Dense, Dropout, GlobalAvgPool1D, GlobalMaxPool1D, Input,
    SpatialDropout1D
)
from tensorflow.python.keras.models import Model as TFModel
from tools import log


class AttentionGru(Classifier):
    def __init__(self):
        # clear_session()
        log('init AttentionGru')
        self.max_length = None
        self._model = None
        self.is_init = False

    def preprocess_data(self, x):
        # if IS_CUT_AUDIO:
        #     x = [sample[0:MAX_AUDIO_DURATION*AUDIO_SAMPLE_RATE] for sample in x]
        # extract mfcc
        x = extract_mfcc_parallel(x,
                                  sr=self.model_config["common"]["sr"],
                                  fft_duration=self.model_config["common"]["fft_duration"],
                                  hop_duration=self.model_config["common"]["hop_duration"],
                                  n_mfcc=self.model_config["common"]["num_mfcc"])
        if self.max_length is None:
            self.max_length = get_max_length(x)
            self.max_length = min(self.model_config["common"]["max_frame_num"], self.max_length)
        x = pad_seq(x, pad_len=self.max_length)
        return x

    def init_model(self, input_shape, num_classes, model_config, **kwargs):
        inputs = Input(shape=input_shape)
        # bnorm_1 = BatchNormalization(axis=-1)(inputs)
        x = Bidirectional(CuDNNLSTM(96, name='blstm1', return_sequences=True),
                          merge_mode='concat')(inputs)
        # activation_1 = Activation('tanh')(lstm_1)
        x = SpatialDropout1D(0.1)(x)
        x = Attention(8, 16)([x, x, x])
        x1 = GlobalMaxPool1D()(x)
        x2 = GlobalAvgPool1D()(x)
        x = Concatenate(axis=-1)([x1, x2])
        x = Dense(units=128, activation='elu')(x)
        x = Dense(units=64, activation='elu')(x)
        x = Dropout(rate=0.4)(x)
        outputs = Dense(units=num_classes, activation='softmax')(x)

        self.model_config = model_config

        model = TFModel(inputs=inputs, outputs=outputs)
        optimizer = optimizers.Adam(
            # learning_rate=1e-3,
            lr = self.model_config["optimizer"]["lr_attention_gru"],
            beta_1 = 1-self.model_config["optimizer"]["beta_1"],
            beta_2 = 1-self.model_config["optimizer"]["beta_2"],
            epsilon = self.model_config["optimizer"]["epsilon"],
            decay = self.model_config["optimizer"]["decay"],
            amsgrad = self.model_config["optimizer"]["amsgrad"]
        )
        model.compile(
            optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy']
        )
        model.summary()
        self._model = model
        self.is_init = True

    def fit(self, train_x, train_y, validation_data_fit, round_num, **kwargs):
        val_x, val_y = validation_data_fit
        if round_num >= 2:
            epochs = 10
        else:
            epochs = 5
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
