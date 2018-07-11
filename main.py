# -*- coding: utf-8 -*-
from data import *
from model import NerModel
from keras.models import load_model
from keras_contrib.layers import CRF



def create_custom_objects():
    instanceHolder = {"instance": None}
    class ClassWrapper(CRF):
        def __init__(self, *args, **kwargs):
            instanceHolder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)
    def loss(*args):
        method = getattr(instanceHolder["instance"], "loss_function")
        return method(*args)
    def accuracy(*args):
        method = getattr(instanceHolder["instance"], "accuracy")
        return method(*args)
    return {"ClassWrapper": ClassWrapper ,"CRF": ClassWrapper, "loss": loss, "accuracy":accuracy}

def train():
    train_data, train_label, word2id, word_embedding, max_sentence_len,tag2id = load_all("data/train_data", "data/vocab.pkl",
                                                                                  "data/vocab_embedding.pkl")
    ner_model = NerModel(word2id,tag2id, word_embedding, tags, max_sentence_len, 300)
    ner_model.train(train_data, train_label, save_path="model/ner.h5")

def predict(sentences):
    sentences_list = list(map(lambda x: list(x), sentences))
    word2id, tag2id=load_base("data/vocab.pkl")
    sents,_=convert_sentenct(sentences_list,100,word2id,tag2id)
    model = load_model('model/ner.h5',custom_objects=create_custom_objects())
    res=model.predict(sents)
    PERs,LOCs,ORGs=getEntity(sentences_list,res)
    for i,sent in enumerate(sentences):
        print(sent)
        print("PER: ",PERs[i])
        print("LOC: ",LOCs[i])
        print("ORG: ",ORGs[i])


if __name__ == '__main__':
    # train()
    sentences=["几年前，计春华曾来成都拍戏。应著名女企业家、红旗连锁股份有限公司董事长曹世如女士的邀请"]
    predict(sentences)