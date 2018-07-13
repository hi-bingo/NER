# -*- coding: utf-8 -*-
from data import *
from model import NerModel
from keras.models import load_model
from keras_contrib.layers import CRF
from sklearn.metrics import classification_report
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import settings
import argparse




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
    train_data, train_label, word2id, word_embedding, max_sentence_len = load_all(settings.TRAIN_PATH,settings.VOCAB_PATH,
                                                                                  settings.VOCAB_EMBEDDING_PATH)
    word_embedding=np.random.uniform(-0.25,0.25,word_embedding.shape)
    ner_model = NerModel(word2id, word_embedding, settings.TAGS, max_sentence_len, settings.EMBEDDING_SIZE)
    ner_model.train(train_data, train_label, save_path=settings.MODEL_PATH)

def predict(sentences,model_max_len=100):
    sentences_list = list(map(lambda x: list(x), sentences))
    word2id=load_word2id(settings.VOCAB_PATH)
    sents,_=convert_sentences(sentences_list,np.array([[]]),model_max_len,word2id)

    res=model.predict(sents)
    PERs,LOCs,ORGs=getEntity(sentences_list,res)
    for i,sent in enumerate(sentences):
        print("##################################")
        print(sent)
        print("PER: ",PERs[i])
        print("LOC: ",LOCs[i])
        print("ORG: ",ORGs[i])



def eval(test_path,model_max_len=100):
    data = read_data(test_path)
    word2id = load_word2id(settings.VOCAB_PATH)
    sents, labels = convert_sentences(np.array(data)[:,0], np.array(data)[:,1],model_max_len, word2id)
    model = load_model(settings.MODEL_PATH, custom_objects=create_custom_objects())
    labels_pred = model.predict(sents)

    tags_pred=[list(map(lambda i: settings.TAGS[i], np.argmax(label, axis=1))) for label in labels_pred]
    # y_true=[tag.split('-')[-1] for tag in np.array(labels).flatten()]
    # y_pred = [tag.split('-')[-1] for tag in np.array(tags_pred).flatten()]
    # y_true = [tag for tag in np.array(labels).flatten() ]
    # y_true=list(map( lambda tag: tag if not tag =="0" else "O" ,np.array(labels).flatten()))
    # y_pred = [tag for tag in np.array(tags_pred).flatten()]
    y_true=list(map( lambda tags: [tag if not tag =="0" else "O" for tag in tags] ,labels))
    y_pred=tags_pred
    metric(y_true,y_pred)


def metric(y_true, y_pred):

    labels=settings.TAGS.copy()
    labels.remove('O')  # remove 'O' label from evaluation
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))  # group B and I results
    print(sklearn_crfsuite.metrics.flat_classification_report(y_true, y_pred, labels=sorted_labels, digits=3))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='train',help="mode: train/eval/demo ")
    args = parser.parse_args()
    if args.m=="train":
        train()
    elif  args.m=="eval":
        model = load_model(settings.MODEL_PATH, custom_objects=create_custom_objects())
        eval(settings.TEST_PATH)
    elif args.m=="demo":
        model = load_model(settings.MODEL_PATH, custom_objects=create_custom_objects())
        while (1):
            print('Input sentence:')
            sent = input()
            if sent == '' or sent.isspace():
                print('Exit...')
                break
            else:
                sentences = [sent]
                predict(sentences, model_max_len=100)

    else:
        print("parameter error")
