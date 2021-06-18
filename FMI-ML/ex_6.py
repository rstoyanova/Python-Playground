#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################

### Упражнение 6
###
### За да работи програмата трябва корпуса от публицистични текстове за Югоизточна Европа,
### да се намира разархивиран в директорията, в която е програмата (виж упражнение 2).

### Преди да се стартира програмата е необходимо да се активира съответното обкръжение с командата:
###
### conda activate tii
###
###
### ВАЖНО!!!
### Настоящата програма използва библиотеката sklearn
### За да я инсталирате, след активиране на обкръжението трябва да изпълните командата:
###
### conda install scikit-learn
###



import nltk
from nltk.corpus import PlaintextCorpusReader
import sys
import random
import math
import pprint

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD

class progressBar:
    def __init__(self ,barWidth = 50):
        self.barWidth = barWidth
        self.period = None
    def start(self, count):
        self.item=0
        self.period = int(count / self.barWidth)
        sys.stdout.write("["+(" " * self.barWidth)+"]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth+1))
    def tick(self):
        if self.item>0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1
    def stop(self):
        sys.stdout.write("]\n")

def extractDictionary(corpus, limit=20000):
    pb = progressBar()
    pb.start(len(corpus))
    dictionary = {}
    for doc in corpus:
        pb.tick()
        for w in doc:
            if w not in dictionary: dictionary[w] = 0
        dictionary[w] += 1
    L = sorted([(w,dictionary[w]) for w in dictionary], key = lambda x: x[1] , reverse=True)
    if limit > len(L): limit = len(L)
    words = [ w for w,_ in L[:limit] ]
    word2ind = { w:i for i,w in enumerate(words)}
    pb.stop()
    return words, word2ind


def co_occurrence_matrix(corpus, window_size=4, limit=20000):
    words, word2ind = extractDictionary(corpus,limit=limit)
    num_words = len(words)
    
    X=np.zeros((num_words,num_words))
    
    pb = progressBar()
    pb.start(len(corpus))
    for doc in corpus:
        pb.tick()
        for wi in range(len(doc)):
            if doc[wi] not in word2ind: continue
            i=word2ind[doc[wi]]
            for k in range(1,window_size+1):
                if wi-k>=0 and doc[wi-k] in word2ind:
                    j=word2ind[doc[wi-k]]
                    X[i,j] += 1
                if wi+k<len(doc) and doc[wi+k] in word2ind:
                    j=word2ind[doc[wi+k]]
                    X[i,j] += 1
    pb.stop()
    return X, words, word2ind

def PMI_matrix(C):
    rowSums = np.sum(C,axis=1)
    colSums = np.sum(C,axis=0)
    D = np.sum(rowSums)
    Z = np.outer(rowSums,colSums)
    X = np.maximum(np.log( D * C / Z),0)
    return X

def PCA_k_dim(X, k=2):
    # Документация на метода има на https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html

    print("Running Incremental PCA over %i words..." % (X.shape[0]))
    X0 = X - np.mean(X, axis=0)
    pca = IncrementalPCA(n_components=k)
    pca.fit(X0)
    X_reduced = pca.transform(X0)
    print("Done.")
    return X_reduced

def SVD_k_dim(X, k=100, n_iters = 10):
    # Документация на метода има на https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
    print("Running Truncated SVD over %i words..." % (X.shape[0]))
    svd = TruncatedSVD(n_components=k, n_iter=n_iters)
    svd.fit(X)
    X_reduced = svd.transform(X)
    print("Done.")
    return X_reduced

def plot_embeddings(M, word2ind, words):
    xs = M[:,0]
    ys = M[:,1]
    for w in words:
        i=word2ind[w]
        plt.scatter(xs[i],ys[i], marker='x', color= 'red')
        plt.text(xs[i]+0.001, ys[i]+0.001, w)
    plt.show()

def plot_embeddings_3d(M, word2ind, words):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = M[:,0]
    ys = M[:,1]
    zs = M[:,2]
    for w in words:
        i=word2ind[w]
        ax.scatter(xs[i], ys[i], zs[i], marker='x', color= 'red')
        ax.text(xs[i]+0.001, ys[i]+0.001, zs[i]+0.001, w)
    plt.show()


#############################################################
#######   Малък пример
#############################################################

text = '''
    две хубави очи душата на дете
    в две хубави очи музика лъчи
    не искат и не обещават те
    душата ми се моли
    дете
    душата ми се моли
    страсти и неволи
    ще хвърлят утре върху тях
    булото на срам и грях
    булото на срам и грях
    не ще го хвърлят върху тях
    страсти и неволи
    душата ми се моли
    дете
    душата ми се моли
    не искат и не обещават те
    две хубави очи музика лъчи
    в две хубави очи душата на дете
    '''

CC,ccww,ccwi = co_occurrence_matrix([text.split()], window_size=1, limit=10)
CC_reduced =SVD_k_dim(CC,k=3)
plot_embeddings(CC_reduced, ccwi, ccww)

# Документация за np.linalg.norm има на https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
CC_lengths = np.linalg.norm(CC_reduced, axis=1)[:, np.newaxis]
CC_normalized = CC_reduced / CC_lengths
plot_embeddings_3d(CC_normalized, ccwi, ccww)


#############################################################
#######   Пълен пример
#############################################################

corpus_root = 'JOURNALISM.BG/C-MassMedia'
myCorpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
startToken = '<START>'
endToken = '<END>'
corpus =  [[startToken] + [w.lower() for w in myCorpus.words(f)] + [endToken] for f in myCorpus.fileids()]

C, words, word2ind = co_occurrence_matrix(corpus)
C_reduced = SVD_k_dim(C)

C_reduced_2d = C_reduced[:,:2]
C_lengths_2d = np.linalg.norm(C_reduced_2d, axis=1)
C_normalized_2d = C_reduced_2d / C_lengths_2d[:, np.newaxis]

sampleWords = ['кола', 'автомобил', 'румъния', 'министър', 'президент', 'гърция', 'футбол', 'спорт', 'баскетбол', 'българия', 'театър', 'кино', 'опера']

plot_embeddings(C_normalized_2d, word2ind, sampleWords)

C_reduced_3d = C_reduced[:,:3]
C_lengths_3d = np.linalg.norm(C_reduced_3d, axis=1)
C_normalized_3d = C_reduced_3d / C_lengths_3d[:, np.newaxis]

plot_embeddings_3d(C_normalized_3d, word2ind, sampleWords)

X = PMI_matrix(C)
X_reduced = SVD_k_dim(X)
X_reduced_3d = X_reduced[:,:3]
X_lengths_3d = np.linalg.norm(X_reduced_3d, axis=1)
X_normalized_3d = X_reduced_3d / X_lengths_3d[:, np.newaxis]
plot_embeddings_3d(X_normalized_3d, word2ind, sampleWords)

C_lengths_100d = np.linalg.norm(C_reduced, axis=1)
C_normalized_100d = C_reduced / C_lengths_100d[:, np.newaxis]

X_lengths_100d = np.linalg.norm(X_reduced, axis=1)
X_normalized_100d = X_reduced / X_lengths_100d[:, np.newaxis]

def most_similar(w,word2ind,words,C,limit=10):
    i = word2ind[w]
    L = np.dot(C,C[i]).tolist()
    L = sorted([(words[i],s) for i,s in enumerate(L)], key = lambda x: x[1] , reverse=True)
    return L[:limit]

pprint.pprint(most_similar('гърция',word2ind,words,C_normalized_100d))
pprint.pprint(most_similar('футбол',word2ind,words,C_normalized_100d))
pprint.pprint(most_similar('камион',word2ind,words,C_normalized_100d))

pprint.pprint(most_similar('гърция',word2ind,words,X_normalized_100d))
pprint.pprint(most_similar('футбол',word2ind,words,X_normalized_100d))
pprint.pprint(most_similar('камион',word2ind,words,X_normalized_100d))

