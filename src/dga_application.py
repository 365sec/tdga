from pandas.io.parsers import read_csv
import csv
from optparse import OptionParser
from dga_detect2 import *

def detect_100(Webnames,features,methods):
    outcome = [Webnames]
    sum_of_average = []
    for feature in features:
        for method in methods:
            if feature == '2-gram' and method == 'kmeans':
                break
            if feature == '234-gram' and method != 'mlp':
                break
            if feature == 'charseq' and method != "rnn":
                break
            dga_test = dga_dect(feature, method)

            #As k-means method do not have the probability that a sample belongs to a class, we should not include it when calculusing the average of prob
            if method != 'kmeans' and method != 'svm':
                dga_prob = dga_test.predict(Webnames)
                outcome = outcome + [dga_prob]
                sum_of_average = sum_of_average +[dga_prob]
            else:
                dga_outcome =dga_test.predict(Webnames)
                outcome = outcome + [dga_outcome]
    average = np.mean(sum_of_average, 0)
    outcome = outcome + [average]
    outcome = np.transpose(outcome)
    return outcome

parser = OptionParser()
parser.add_option("-i", "--input",dest="input", action="store", type="string",
                   help="input file for predict dga domains" )
parser.add_option("-o", "--output", dest="output" ,action="store", type="string",
                   help='output file for dga domains')
options, args = parser.parse_args()


features = [ 'charseq','textfeature']
methods = ['mlp','svm','nb','kmeans','xgboost']



inputfilepath = options.input
outputfilepath = options.output
with open(outputfilepath, 'w') as dga_detect_outcome:
    filewriter = csv.writer(dga_detect_outcome, delimiter=',',
                     quoting=csv.QUOTE_MINIMAL)
    #Determine the name of the column
    names = ['Webname']
    for feature in features:
        for method in methods:
            if feature == '2-gram' and method == 'kmeans':
                break
            if feature == '234-gram' and method != 'mlp':
                break
            if feature == 'charseq' and method != "rnn":
                break
            name = feature + '&' + method
            names = names + [name]
    names = names + ['average of prob']
    filewriter.writerow(names)

    fp = open(inputfilepath,"rb")
    Webnames = []
    k = 0
    for line in fp:
        if len(Webnames)>100:
            k = k + 1
            outcome_100 = detect_100(Webnames, features, methods)
            for row in outcome_100:
                filewriter.writerow(row)
            Webnames = []
            if k == 100:
                os.exit()
        Webnames.append(line)
    outcome_100 = detect_100(Webnames, features, methods)
    for row in outcome_100:
        filewriter.writerow(row)
