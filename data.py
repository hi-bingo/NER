# -*- coding: utf-8 -*-

import codecs,pickle
import numpy as np



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
            word_embedding.append(np.random.uniform(-0.25, 0.25,embedding_size))
    print("random embedding num {}".format(not_found))
    with codecs.open(embedding_save, "wb") as f:
        pickle.dump(word_embedding, f)
    return embedding

if __name__ == '__main__':
    # data=read_data("data/train_data")
    # embedding=load_word2vec("data/zh_wiki.vec")
    em=get_vocab("data/train_data","data/zh_wiki.vec","data/vocab.pkl","data/vocab_embedding.pkl",embedding_size=300)
