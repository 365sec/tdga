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
from pandas.io.parsers import read_csv
import csv
from optparse import OptionParser



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

        CV = CountVectorizer(
                                        ngram_range=(2, 2),
                                        token_pattern=r'\w',
                                        decode_error='ignore',
                                        strip_accents='ascii',
                                        max_features=10000,
                                        stop_words='english',
                                        max_df=1.0,
                                        min_df=1)
        x = CV.fit_transform(domains)
        return x.toarray()


    def get_feature_234gram(self, domains):

        CV = CountVectorizer(
                                        ngram_range=(2, 4),
                                        token_pattern=r'\w',
                                        decode_error='ignore',
                                        strip_accents='ascii',
                                        max_features=max_features,
                                        stop_words='english',
                                        max_df=1.0,
                                        min_df=1)
        x = CV.fit_transform(domains)
        return x.toarray()

    def predict_Webnames(self, Webnames, filewriter):
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


        #load trained model
        model_name = self.feature + '&' + self.method + '.model'
        model = joblib.load(model_name)

        #based on the known modles, training data
        if self.method == 'kmeans':
            label = pd.read_csv('../data/kmeans_type.csv')
            labeltype = label.values[0][0]
            print labeltype==1
            if labeltype == 1:
                y_pred = model.predict(x)
            else:
                if labeltype == 2:
                    y_pred1 = model.predict(x)
                    y_pred = []
                    for elem in y_pred1:
                        y_pred = y_pred + [1-elem]
                else:
                    print "Error: No fitted label character for kmeans"
                    os._exit(0)
        else:
            y_pred = model.predict(x)
            y_pred_prob = model.predict_proba(x)

        #Transform outcome '0'&'1' into more comprehesive form 'Alexa'&'DGA'
        pred = []
        for elem in y_pred:
            if elem == 0:
                pred = pred + ['Alexa']
            else:
                pred = pred + ['DGA']

        #Calculus the probability of being DGA
        pred_prob = np.transpose(y_pred_prob)[1]

        #Write the prediction in to the outcome file
        outcome = [Webnames,pred,pred_prob]
        rows = np.transpose(outcome)
        for row in rows:
            filewriter.writerow(row)



    def predict(self, inputfilepath, outputfilepath):

        #First create an empty outcome file with only the column names
        os.remove(outputfilepath)
        with open(outputfilepath) as dga_detect_outcome:
            filewriter = csv.writer(dga_detect_outcome, delimiter=',',
                             quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['Webname', 'DGA or Alexa','Prob of DGA'])

            #Update the outcome file using 'predict_domains' function every 100 domains
            fp = open(inputfilepath,"rb")
            Webnames = []
            for line in fp:
                if len(Webnames) > 100:
                    self.predict_Webnames(Webnames, filewriter)
                    os.exit()
                    Webnames=[]
                Webnames.append(line)
            self.predict_Webnames(Webnames, filewriter)



if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-i", "--input",dest="input", action="store", type="string",
                       help="input file for predict dga domains" )
    parser.add_option("-o", "--output", dest="output" ,action="store", type="string",
                       help='output file for dga domains')
    parser.add_option('-f', '--feature', dest='feature', type='choice',
                       choices=['2-gram', '234-gram', 'charseq'], default=None,
                       help='feature ....')
    parser.add_option('-m', '--method', dest='method', type='choice',
                       choices=['kmeans', 'mlp', 'rnn','svm'], default=None,
                       help='method ....')
    options, args = parser.parse_args()


    dga_test = dga_dect(options.feature, options.method)
    dga_test.predict(options.input, options.output)
    outcome = pd.read_csv(options.output, header = True, sep = ',')
    print sample(outcome, 100)

    '''
    dga_test = dga_dect('textfeature', 'nb')
    dgapath = "../data/domain.txt"
    dga_test.predict(dgapath)
    '''
