#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2019/9/27 10:12
# @Author:  Mecthew
import tensorflow as tf
from data_process import extract_mfcc_parallel, get_max_length, ohe2cat, pad_seq
from models.attention import Attention
from models.my_classifier import Classifier
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.layers import (
    Activation, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPool1D, Input, SpatialDropout1D
)
from tensorflow.python.keras.models import Model as TFModel
from tools import log


class BilstmAttention(Classifier):
    def __init__(self):
        # clear_session()
        log('init BilstmAttention')
        self.max_length = None
        self._model = None
        self.is_init = False

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
            self.max_length = min(800, self.max_length)
        x = pad_seq(x, pad_len=self.max_length)
        return x

    def init_model(self, input_shape, num_classes, model_config, **kwargs):
        inputs = Input(shape=input_shape)
        # bnorm_1 = BatchNormalization(axis=2)(inputs)
        lstm_1 = Bidirectional(
            LSTM(64, name='blstm_1', return_sequences=True), merge_mode='concat'
        )(inputs)
        activation_1 = Activation('tanh')(lstm_1)
        dropout1 = SpatialDropout1D(0.5)(activation_1)
        attention_1 = Attention(8, 16)([dropout1, dropout1, dropout1])
        pool_1 = GlobalMaxPool1D()(attention_1)
        dropout2 = Dropout(rate=0.5)(pool_1)
        dense_1 = Dense(units=256, activation='relu')(dropout2)
        outputs = Dense(units=num_classes, activation='softmax')(dense_1)

        self.model_config = model_config

        model = TFModel(inputs=inputs, outputs=outputs)
        optimizer = optimizers.Adam(
            lr = self.model_config["optimizer"]["lr_bilstm_attention"],
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
