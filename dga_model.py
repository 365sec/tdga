import xgboost as xgb
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.neural_network import MLPClassifier
from tflearn.layers.normalization import local_response_normalization
from tensorflow.contrib import learn
import gensim
import re
from collections import namedtuple
from random import sample
from random import shuffle
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from hmmlearn import hmm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from sklearn.externals import joblib
from pandas.core.frame import DataFrame


random_state = 170

dga_file="../data/dga/dga.txt"
alexa_file="../data/dga/top-1m.csv"

def load_alexa():
    x=[]
    data = pd.read_csv(alexa_file, sep=",",header=None)
    x=[i[1] for i in data.values]
    domain = []
    k = 0
    for net in x:
        k = k + 1
        domain_name = ''
        for elem in net:
            if elem != '.':
                domain_name = domain_name + elem
            else:
                break
        domain = domain + [domain_name]
        if k > 10000:
            break
    return domain

def load_dga():
    x=[]
    data = pd.read_csv(dga_file, sep="\t", header=None,
                       skiprows=18)
    x=[i[1] for i in data.values]
    domain = sample(x,10000)
    return domain

def get_feature_charseq():
    alexa=load_alexa()
    dga=load_dga()
    x=alexa+dga
    max_features=10000
    y=[0]*len(alexa)+[1]*len(dga)

    t=[]
    for i in x:
        v=[]
        for j in range(0,len(i)):
            v.append(ord(i[j]))
        t.append(v)

    x=t

    return x, y


def get_aeiou(domain):
    count = len(re.findall(r'[aeiou]', domain.lower()))
    #count = (0.0 + count) / len(domain)
    return count

def get_uniq_char_num(domain):
    count=len(set(domain))
    #count=(0.0+count)/len(domain)
    return count

def get_uniq_num_num(domain):
    count = len(re.findall(r'[1234567890]', domain.lower()))
    #count = (0.0 + count) / len(domain)
    return count

def get_feature():
    from sklearn import preprocessing
    alexa=load_alexa()
    dga=load_dga()
    v=alexa+dga
    y=[0]*len(alexa)+[1]*len(dga)
    x=[]

    for vv in v:
        vvv=[get_aeiou(vv),get_uniq_char_num(vv),get_uniq_num_num(vv),len(vv)]
        x.append(vvv)

    x=preprocessing.scale(x)
    return x, y

def get_feature_2gram():
    alexa=load_alexa()
    dga=load_dga()
    x=alexa+dga
    max_features=10000
    y=[0]*len(alexa)+[1]*len(dga)

    CV = CountVectorizer(
                                    ngram_range=(2, 2),
                                    token_pattern=r'\w',
                                    decode_error='ignore',
                                    strip_accents='ascii',
                                    max_features=max_features,
                                    stop_words='english',
                                    max_df=1.0,
                                    min_df=1)
    x = CV.fit_transform(x)
    return x.toarray(), y


def get_feature_234gram():
    alexa=load_alexa()
    dga=load_dga()
    x=alexa+dga
    max_features=10000
    y=[0]*len(alexa)+[1]*len(dga)

    CV = CountVectorizer(
                                    ngram_range=(2, 4),
                                    token_pattern=r'\w',
                                    decode_error='ignore',
                                    strip_accents='ascii',
                                    max_features=max_features,
                                    stop_words='english',
                                    max_df=1.0,
                                    min_df=1)
    x = CV.fit_transform(x)
    return x.toarray(), y

def do_nb(x, y):
    gnb = GaussianNB()
    gnb.fit(x, y)
    return gnb

def do_xgboost(x, y):
    xgb_model = xgb.XGBClassifier().fit(x, y)
    return xgb_model

def do_mlp(x, y):

    global max_features
    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    clf.fit(x, y)
    return clf

def do_rnn(trainX, trainY):
    max_document_length=64
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    # Network building
    net = tflearn.input_data([None, max_document_length])
    net = tflearn.embedding(net, input_dim=10240000, output_dim=64)
    net = tflearn.lstm(net, 64, dropout=0.1)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0,tensorboard_dir="dga_log")
    model.fit(trainX, trainY, show_metric=True,
              batch_size=10,run_id="dga",n_epoch=1)

    return model
    

def do_SVM(x, y):
    clf = svm.SVC(kernel='rbf', C=1).fit(x, y)
    return clf

def do_kmeans(x, y):
    model=KMeans(n_clusters=2, random_state=random_state)
    model.fit(x)
    y_pred1 = model.predict(x)
    y_pred2 = []
    for elem in y_pred1:
        y_pred2 = y_pred2 + [1-elem]
    score1 = accuracy_score(y, y_pred1)
    score2 = accuracy_score(y, y_pred2)
    if score1 > score2:
        type = pd.DataFrame([[1]])
        type.to_csv("../data/dga/kmeans_type.csv",index=False)
        return model
    else:
        type = pd.DataFrame([[2]])
        type.to_csv("../data/dga/kmeans_type.csv",index=False)
        return model
    
if __name__ == "__main__":
    
    print "Hello dga"
    print "234-gram & mlp"
    x, y = get_feature_234gram()
    model1 = do_mlp(x, y)
    joblib.dump(model1,'234-gram&mlp.model')
    
    print "text feature & nb"
    x, y = get_feature()
    model2 = do_nb(x, y)
    joblib.dump(model2, 'textfeature&nb.model')
    
    print "text feature & xgboost"
    x, y = get_feature()
    model3 = do_xgboost(x, y)
    joblib.dump(model3, 'textfeature&xgboost.model')
    
    print "text feature & mlp"
    x, y = get_feature()
    model4 = do_mlp(x, y)
    joblib.dump(model4, 'textfeature&mlp.model')
    
    print "charseq & rnn"
    x, y = get_feature_charseq()
    model5 = do_rnn(x, y)
    joblib.dump(model5, 'charseq&rnn.model')
    
    print "2-gram & mlp"
    x, y = get_feature_2gram()
    model6 = do_mlp(x, y)
    joblib.dump(model6, '2-gram&mlp.model')
    
    print "2-gram & XGBoost"
    x, y = get_feature_2gram()
    model7 = do_xgboost(x, y)
    joblib.dump(model7, '2-gram&xgboost.model')
    
    print "2-gram & nb"
    x, y=get_feature_2gram()
    model8 = do_nb(x, y)
    joblib.dump(model8, '2-gram&nb.model')
    
    print "2-gram & SVM"
    x, y=get_feature_2gram()
    model9 = do_SVM(x, y)
    joblib.dump(model9, '2-gram&svm.model')
    
    print "text feature & svm"
    x, y = get_feature()
    model10 = do_SVM(x, y)
    joblib.dump(model10, 'textfeature&svm.model')
    
    print "text feature & kmeans"
    x, y = get_feature()
    model11 = do_kmeans(x, y)
    joblib.dump(model11, 'textfeature&kmeans.model')
    
    
  