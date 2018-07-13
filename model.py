# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM,Dropout,TimeDistributed,Dense
from keras_contrib.layers import CRF
from sklearn.model_selection import train_test_split
import pickle,settings
import numpy as np
from data import pad_sentence

class NerModel:
    def __init__(self,word2id,word_embedding,tags,max_sentence_len,embedding_size):
        self.word2id=word2id
        self.word_embedding=word_embedding
        self.tags=tags
        self.max_sentence_len=max_sentence_len
        self.embedding_size = embedding_size
        self.lstm_unit=settings.LSTM_UNIT
        self.batch_size = settings.BATCH_SIZE
        self.epoch_num = settings.EPOCH_NUM
        self.dropout_prob=settings.DROP_PROP
        self.lr=settings.LEARNING_RATE
        self.optimizer="adam"

        self.model=self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Embedding(len(self.word2id),
                            self.embedding_size,
                            weights=[self.word_embedding],
                            input_length=self.max_sentence_len,
                            ))
        model.add(Bidirectional(LSTM(self.lstm_unit, return_sequences=True)))
        model.add(Dropout(self.dropout_prob))
        model.add(TimeDistributed(Dense(len(self.tags))))
        crf = CRF(len(self.tags), sparse_target=True)
        model.add(crf)
        model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()
        return model




    def train(self,train_data,train_label,save_path):
        train_data,train_label=pad_sentence(train_data,train_label,self.max_sentence_len,self.word2id)
        train_label = np.expand_dims(train_label, 2)
        X_train, X_valid, y_train, y_vaild = train_test_split(train_data, train_label, test_size = 0.2, random_state = 42)

        callback=keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=2,
                                      verbose=0, mode='auto')
        self.model.fit(X_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epoch_num,
                  validation_data=[X_valid, y_vaild],
                  callbacks=[callback])

        self.model.save(save_path)
