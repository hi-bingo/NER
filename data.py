# -*- coding: utf-8 -*-

import codecs,pickle
import numpy as np
import settings

# tags=['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]
#
# word2id=None
# tag2id=None

def read_data(path):
    '''
    read train data
    :param path: 
    :return: array of sentences,labels 
    '''
    data = []
    with codecs.open(path, "r", "utf-8") as f:
        lines = f.readlines()
    sentence, label= [], []
    for line in lines:
        if line != '\r\n':
            [char, tag] = line.strip().split()
            sentence.append(char)
            label.append(tag)
        else:
            data.append((sentence, label))
            sentence, label= [], []

    return data


def load_word2vec(path,size=300):
    '''
    load word2vec embedding
    :param path: 
    :return: dict of word2vec
    '''
    data = {}
    with codecs.open(path, "r", "utf-8") as f:
        lines = f.readlines()
    for line in lines:
        l=line.strip().split()
        if not len(l)==size+1:
            print("format error")
            continue
        data[l[0]]=[float(c) for c in l[1:]]

    return data




def get_vocab(train_path,embedding_path,vocab_save,embedding_save,embedding_size=300):
    '''
    build train vocabulary
    :param train_path: 
    :param embedding_path: 
    :param embedding_size: 
    :return: 
    '''
    data = read_data(train_path)
    embedding=load_word2vec(embedding_path,embedding_size)
    word2id = dict()
    word_embedding=[]
    for sentence, label in data:
        for word in sentence:
            if word.encode("UTF-8").isdigit():
                word = '<NUM>'
            elif word.encode("UTF-8").isalpha():
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = len(word2id)

    word2id['<UNK>'] = len(word2id)
    word2id['<PAD>'] = len(word2id)

    print(len(word2id))
    with codecs.open(vocab_save, "wb") as f:
        pickle.dump(word2id, f)

    not_found=0
    for word in word2id.keys():
        if word in embedding:
            word_embedding.append(np.array(embedding[word]))
        else:
            not_found+=1
            word_embedding.append(np.zeros(embedding_size))
    print("random embedding num {}".format(not_found))
    with codecs.open(embedding_save, "wb") as f:
        pickle.dump(word_embedding, f)
    return word2id,word_embedding



def parse_sentence(sentenct,word2id):
    '''
    parse senence to list of id
    :param sentenct: 
    :param word2id: 
    :return: 
    '''
    res = []
    for word in sentenct:
        if word.encode("UTF-8").isdigit():
            word = '<NUM>'
        elif word.encode("UTF-8").isalpha():
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        res.append(word2id[word])
    return np.array(res)



def load_word2id(vocab_path):
    word2id = pickle.load(open(vocab_path, "rb"))
    return word2id


def load_all(train_path,vocab_path,word_embedding_path):
    '''
    load all data
    :param train_path: 
    :param vocab_path: 
    :param word_embedding_path: 
    :return: 
    '''
    train_data=[]
    train_label=[]
    word2id=load_word2id(vocab_path)
    word_embedding=pickle.load(open(word_embedding_path, "rb"))
    data=read_data(train_path)
    max_sentence_len=0
    for sentence,label in data:
        if len(sentence)>max_sentence_len:
            max_sentence_len=len(sentence)
        train_data.append(parse_sentence(sentence,word2id))
        train_label.append(np.array([settings.TAG2ID[t] for t in label]))
    return train_data,train_label,word2id,np.array(word_embedding),max_sentence_len

def pad_sentence(train_data,train_label,max_sentence_len,word2id):
    for i in range(len(train_data)):
        if len(train_data[i]) < max_sentence_len:
            train_data[i] = np.append(train_data[i], [word2id["<PAD>"]] * (max_sentence_len - len(train_data[i])))
            train_label[i] = np.append(train_label[i], [settings.TAG2ID["O"]] * (max_sentence_len - len(train_label[i])))
    train_data=np.array(train_data)
    train_label = np.array(train_label)

    return train_data,train_label


def convert_sentences(sentences,labels,max_sentenct_len,word2id):
    sen=[]
    for s in sentences:
        sen.append(parse_sentence(s, word2id))
    return pad_sentence(sen,labels.tolist(),max_sentenct_len,word2id)

def getEntity(sentences,labels):
    PERs, LOCs, ORGs = [], [], []
    for sent,label in zip(sentences,labels):
        PER,LOC,ORG = [],[],[]
        tag=list(map(lambda i:settings.TAGS[i], np.argmax(label,axis=1)))
        i=0
        while i<len(tag):
            if tag[i]=="B-PER":
                per=sent[i]
                i+=1
                while i<len(tag) and tag[i]=="I-PER":
                    per+=sent[i]
                    i += 1
                PER.append(per)
            if tag[i] == "B-LOC":
                loc = sent[i]
                i += 1
                while i < len(tag) and tag[i] == "I-LOC":
                    loc += sent[i]
                    i += 1
                LOC.append(loc)
            if tag[i] == "B-ORG":
                org = sent[i]
                i += 1
                while i < len(tag) and tag[i] == "I-ORG":
                    org += sent[i]
                    i += 1
                ORG.append(org)
            i+=1
        PERs.append(PER)
        ORGs.append(ORG)
        LOCs.append(LOC)
    return PERs,LOCs,ORGs



def load_test(test_path):
    data = read_data(train_path)

if __name__ == '__main__':
    get_vocab(settings.TRAIN_PATH,settings.EMBEDDING_PATH,settings.VOCAB_PATH,settings.VOCAB_EMBEDDING_PATH,embedding_size=settings.EMBEDDING_SIZE)
    # train_data, train_label, word2id, word_embedding, max_sentence_len=load_all(settings.TRAIN_PATH,
    #                                                                             settings.VOCAB_PATH,
    #                                                                             settings.VOCAB_EMBEDDING_PATH)
