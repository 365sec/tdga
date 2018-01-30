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
from random import shuffle
from random import sample
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




class dga_dect():
    def __init__(self, feature, method):
        #Report error when we haven't trained given model
        if feature == '2-gram' and method == 'kmeans':
            print "Error: Do not support this method."
            os._exit(0)
        if feature == '234-gram' and method != 'mlp':
            print "Error: Do not support this method."
            os._exit(0)
        if feature == 'charseq' and method != "rnn":
            print "Error: Do not support this method."
            os._exit(0)

        self.feature = feature
        self.method = method
        #load trained model
        model_name = self.feature + '&' + self.method + '.model'
        self.model = joblib.load(model_name)



    def get_feature_charseq(self, domains):
        t=[]
        for i in domains:
            v=[]
            for j in range(0,len(i)):
                v.append(ord(i[j]))
            t.append(v)

        x=t
        return x


    def get_aeiou(self, domain):
        count = len(re.findall(r'[aeiou]', domain.lower()))
        #count = (0.0 + count) / len(domain)
        return count

    def get_uniq_char_num(self, domain):
        count=len(set(domain))
        #count=(0.0+count)/len(domain)
        return count

    def get_uniq_num_num(self, domain):
        count = len(re.findall(r'[1234567890]', domain.lower()))
        #count = (0.0 + count) / len(domain)
        return count

    def get_feature(self, domains):
        from sklearn import preprocessing

        x=[]

        for vv in domains:
            vvv=[self.get_aeiou(vv),self.get_uniq_char_num(vv),self.get_uniq_num_num(vv),len(vv)]
            x.append(vvv)

        x=preprocessing.scale(x)
        return x

    def get_feature_2gram(self, domains):

        name = self.feature + '&' + self.method + '.plk'
        CV = joblib.load(name)
        x = CV.fit_transform(domains)
        return x.toarray()


    def get_feature_234gram(self, domains):

        name = self.feature + '&' + self.method + '.plk'
        CV = joblib.load(name)
        x = CV.fit_transform(domains)
        return x.toarray()

    def predict(self, Webnames):
        #Transform webname into domain form, e.g.:"google.com" --> "google"
        domains = []
        for net in Webnames:
            net = net[:-1]
            domain = ''
            for elem in net:
                if elem != '.':
                    domain = domain + elem
                else:
                    break
            domains = domains + [domain]

        #Characterize the domains
        if  self.feature == 'textfeature':
            x = self.get_feature(domains)
        if self.feature == '2-gram':
            x = self.get_feature_2gram(domains)
        if self.feature == '234-gram':
            x = self.get_feature_234gram(domains)
        if self.feature == 'charseq':
            x = self.get_feature_charseq(domains)




        #based on the known modles, training data
        if self.method == 'kmeans':
            label = pd.read_csv('../data/kmeans_type.csv')
            labeltype = label.values[0][0]
            print labeltype==1
            if labeltype == 1:
                y_pred = self.model.predict(x)
            else:
                if labeltype == 2:
                    y_pred1 = self.model.predict(x)
                    y_pred = []
                    for elem in y_pred1:
                        y_pred = y_pred + [1-elem]
                else:
                    print "Error: No fitted label character for kmeans"
                    os._exit(0)
            pred = []
            for elem in y_pred:
                if elem == 0:
                    pred = pred + ['Alexa']
                else:
                    pred = pred + ['DGA']
            return pred
        if self.method == 'svm':
            y_pred = self.model.predict(x)
            pred = []
            for elem in y_pred:
                if elem == 0:
                    pred = pred + ['Alexa']
                else:
                    pred = pred + ['DGA']
            return pred
        else:
            y_pred_prob = self.model.predict_proba(x)

            #Calculus the probability of being DGA
            pred_prob = np.transpose(y_pred_prob)[1]
            return pred_prob
