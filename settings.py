# -*- coding: utf-8 -*-
import os
TAGS=['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]
TAG2ID={k: v for v, k in enumerate(TAGS)}

# file parameter
MERTIC_TAGS=['PER','LOC','ORG']
MODEL_PATH="model/ner.h5"
VOCAB_PATH="data/vocab.pkl"
TRAIN_PATH="data/train_data"
TEST_PATH="data/test_data"
VOCAB_EMBEDDING_PATH="data/vocab_embedding.pkl"
EMBEDDING_PATH="data/zh_wiki.vec"
EMBEDDING_SIZE=300


# network parameter
LSTM_UNIT=300
BATCH_SIZE=64
EPOCH_NUM=50
DROP_PROP=0.5
LEARNING_RATE=0.001