#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################

### Упражнение 4b
###
### За да работи програмата трябва корпуса от публицистични текстове за Югоизточна Европа,
### да се намира разархивиран в директорията, в която е програмата (виж упражнение 2).

### Преди да се стартира програмата е необходимо да се активира съответното обкръжение с командата:
### conda activate tii

import nltk
from nltk.corpus import PlaintextCorpusReader
import sys

corpus_root = 'JOURNALISM.BG/C-MassMedia'
myCorpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
fileNames = myCorpus.fileids()

class progressBar:
    def __init__(self ,barWidth = 50):
        self.barWidth = barWidth
        self.period = None
    def start(self, count):
        self.period = int(count / self.barWidth)
        sys.stdout.write("["+(" " * self.barWidth)+"]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth+1))
    def tick(self, item):
        if item>0 and item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
    def stop(self):
        sys.stdout.write("]\n")

def printDocuments(myCorpus,fileNames,docList):
    for docID,score in docList:
        text = myCorpus.words(fileNames[docID])
        print('Document ID: '+str(docID)+', relevance: '+str(score))
        for word in text:
            print(word,end=' ')
        print('\n')

import math

def createIndex(myCorpus, fileNames):
    pb = progressBar()
    pb.start(len(fileNames))
    dictionary={}
    for docID, fileName in enumerate(fileNames):
        pb.tick(docID)
        text = myCorpus.words(fileName)
        for token in text:
            if not token.isalpha(): continue
            term = token.lower()
            if term in dictionary:
                (postings, tf) = dictionary[term]
                if postings[-1] < docID:
                    postings.append(docID)
                    tf.append(1)
                else:
                    tf[-1] += 1
            else:
                dictionary[term] = ([docID],[1])
    pb.stop()
    N = len(fileNames)
    pb.start(len(dictionary))
    lengths = [0.0]*N
    t=0
    for term in dictionary:
        pb.tick(t)
        t += 1
        (postings, tf) = dictionary[term]
        df = len(postings)
        idf = math.log(N/df)
        for i, docID in enumerate(postings):
            w = math.log(1+tf[i]) * idf
            lengths[docID] += w ** 2
    for docID in range(N):
        lengths[docID] = math.sqrt(lengths[docID])
    pb.stop()
    return dictionary, lengths

dictionary, lengths = createIndex(myCorpus, fileNames)

def cosineScore(query, K, dictionary, lengths):
    N = len(lengths)
    scores = {}
    for term in query:
        if term not in dictionary: continue
        (postings, tf) = dictionary[term]
        df_t = len(postings)
        idf_t = math.log(N/df_t)
        #        w_tq = idf_t
        w_tq = 1.0
        for i, docID in enumerate(postings):
            w_td = math.log(1+tf[i]) * idf_t
            if docID not in scores: scores[docID]=0.0
            scores[docID] += w_tq * w_td
    for docID in scores:
        scores[docID] = scores[docID] / lengths[docID]
    result = sorted([(docID,scores[docID]) for docID in scores], key = lambda x: x[1] , reverse=True)
    return result[:K]

printDocuments(myCorpus, fileNames, cosineScore(['световно', 'първенство', 'по', 'футбол'], 5, dictionary, lengths))

printDocuments(myCorpus, fileNames, cosineScore(['румъния','вирус'], 5, dictionary, lengths))
