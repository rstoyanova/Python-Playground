#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################

### Упражнение 8
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
from sklearn.decomposition import TruncatedSVD

#############################################################
###  Визуализация на прогреса
#############################################################
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

#############################################################
###  Разбиване на корпус на тестов и тренинг
#############################################################
def splitSentCorpus(fullSentCorpus, testFraction = 0.1):
    random.seed(42)
    random.shuffle(fullSentCorpus)
    testCount = int(len(fullSentCorpus) * testFraction)
    testSentCorpus = fullSentCorpus[:testCount]
    trainSentCorpus = fullSentCorpus[testCount:]
    return testSentCorpus, trainSentCorpus


###################################################################
####
####   Мултиномен Бейсов класификатор от упражнение 4
####
###################################################################

def trainMultinomialNB(trainClassCorpus):
    N = sum(len(classList) for classList in trainClassCorpus)
    classesCount = len(trainClassCorpus)
    pb = progressBar(50)
    pb.start(N)
    V = {}
    for c in range(classesCount):
        for text in trainClassCorpus[c]:
            pb.tick()
            terms = [ token.lower() for token in text if token.isalpha() ]
            for term in terms:
                if term not in V:
                    V[term] = [0] * classesCount
                V[term][c] += 1
    pb.stop()

    Nc = [ (len(classList)) for classList in trainClassCorpus ]
    prior = [ Nc[c] / N for c in range(classesCount) ]
    T = [0] * classesCount
    for t in V:
        for c in range(classesCount):
            T[c] += V[t][c]
    condProb = {}
    for t in V:
        condProb[t] = [ (V[t][c] +1) / (T[c] + len(V)) for c in range(classesCount)]
    return condProb, prior, V

def applyMultinomialNB(prior, condProb, text, features = None ):
    terms = [ token.lower() for token in text if token.isalpha() ]
    for c in range(len(prior)):
        score = math.log(prior[c])
        for t in terms:
            if t not in condProb: continue
            if features and t not in features: continue
            score += math.log(condProb[t][c])
        if c == 0 or score > maxScore:
            maxScore = score
            answer = c
    return answer

def testClassifier(testClassCorpus, gamma):
    L = [ len(c) for c in testClassCorpus ]
    pb = progressBar(50)
    pb.start(sum(L))
    classesCount = len(testClassCorpus)
    confusionMatrix = [ [0] * classesCount for _ in range(classesCount) ]
    for c in range(classesCount):
        for text in testClassCorpus[c]:
            pb.tick()
            c_MAP = gamma(text)
            confusionMatrix[c][c_MAP] += 1
    pb.stop()
    precision = []
    recall = []
    Fscore = []
    for c in range(classesCount):
        extracted = sum(confusionMatrix[x][c] for x in range(classesCount))
        if confusionMatrix[c][c] == 0:
            precision.append(0.0)
            recall.append(0.0)
            Fscore.append(0.0)
        else:
            precision.append( confusionMatrix[c][c] / extracted )
            recall.append( confusionMatrix[c][c] / L[c] )
            Fscore.append((2.0 * precision[c] * recall[c]) / (precision[c] + recall[c]))
    P = sum( L[c] * precision[c] / sum(L) for c in range(classesCount) )
    R = sum( L[c] * recall[c] / sum(L) for c in range(classesCount) )
    F1 = (2*P*R) / (P + R)
    print('=================================================================')
    print('Матрица на обърквания: ')
    for row in confusionMatrix:
        for val in row:
            print('{:4}'.format(val), end = '')
        print()
    print('Прецизност: '+str(precision))
    print('Обхват: '+str(recall))
    print('F-оценка: '+str(Fscore))
    print('Обща презизност: '+str(P))
    print('Общ обхват: '+str(R))
    print('Обща F-оценка: '+str(F1))
    print('=================================================================')
    print()

###################################################################################
####
####   Влагане на думи в нискомерно гъсто векторно пространство от упражнение 6
####
###################################################################################

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

def SVD_k_dim(X, k=100, n_iters = 10):
    # Документация на метода има на https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
    print("Running Truncated SVD over %i words..." % (X.shape[0]))
    svd = TruncatedSVD(n_components=k, n_iter=n_iters)
    svd.fit(X)
    X_reduced = svd.transform(X)
    print("Done.")
    return X_reduced


#############################################################
###  Влагане на документи в нискомерно гъсто векторно пространство

def docVector(document, Embedding, word2ind):
    tf = np.zeros(len(word2ind))
    for w in document:
        if w in word2ind:
            tf[word2ind[w]] += 1
    d=np.dot(tf,Embedding)
    return d / np.linalg.norm(d)

def corpusEmbedding(corpus, Embedding, word2ind):
    return np.stack([ docVector(doc, Embedding, word2ind) for doc in corpus ])

#############################################################
#############################################################
####
####    Логистична регресия -- Бинарен класификатор
####
#############################################################
#############################################################

def sigmoid(x):
    return 1/(1+np.exp(-x))

def crossEntropyS(X, Y, w, b):
    m = X.shape[0]
    s = sigmoid(np.dot(X,w)+b)
    pr = (1-Y) + (2*Y-1)*s
    ce = -np.mean(np.log(pr))
    return ce

def gradCrossEntropyS(X,Y,w,b):
    m = X.shape[0]
    g = Y - sigmoid(np.dot(X,w)+b)
    db = -np.mean(g)
    dw = -np.mean( g[:,np.newaxis] * X,axis=0)
    return dw, db

#############################################################
#############################################################
####
####    Логистична регресия -- класификатор при много класове
####
#############################################################
#############################################################

def softmaxV(u):
    ### u следва да бъде вектор с резмер k
    e = np.exp(u)
    return e / np.sum(e)

def softmaxM(U):
    ### U следва да бъде матрица с размерност: (m,k)
    e = np.exp(U)
    return e / np.sum(e,axis=1)[:,np.newaxis]

def crossEntropyM(X, Y, W, b):
    ### класовете са k
    ### X е с размерност: (m,n)
    ### Y е с размерност: (m)
    ### W е с размерност: (n,k)
    ### b е с размерност: (k)

    m = X.shape[0]
    s = softmaxM(np.dot(X,W)+b[np.newaxis,:])
    pr = s[np.arange(m),Y]
    ce = -np.mean(np.log(pr))
    return ce

def gradCrossEntropyM(X,Y,W,b):
    m = X.shape[0]
    k = W.shape[1]
    s = softmaxM(np.dot(X,W)+b[np.newaxis,:])
    v = -s
    v[np.arange(m),Y] += 1.
    db = -np.mean(v, axis=0)
    dW = -(1/m) * np.dot(X.transpose(), v)
    return dW, db

#############################################################
#############################################################
####
####    спускане по градиента
####
#############################################################
#############################################################

def gradientDescend(X,Y,tX,tY,w0,b0,crossEntropy,gradCrossEntropy,alpha=1.):
    epoch=0
    w=w0
    b=b0
    ceList = []
    tceList = []
    while epoch<100000:
        if epoch % 1000 == 0:
            ce = crossEntropy(X, Y, w, b)
            tce = crossEntropy(tX, tY, w, b)
            print(epoch,ce,tce)
            ceList.append(ce)
            tceList.append(tce)
        epoch += 1
        dw, db = gradCrossEntropy(X,Y,w,b)
        b -= alpha * db
        w -= alpha * dw
    return w,b,ceList,tceList

#############################################################
#######   Зареждане на корпуса
#############################################################

corpus_root = 'JOURNALISM.BG/C-MassMedia'
myCorpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
startToken = '<START>'
endToken = '<END>'
fileNames = myCorpus.fileids()

ecoCorpus = [ [startToken] + [w.lower() for w in myCorpus.words(f)] + [endToken] for f in fileNames if f.find('E-Economy'+'/')==0 ]
milCorpus = [ [startToken] + [w.lower() for w in myCorpus.words(f)] + [endToken] for f in fileNames if f.find('S-Military'+'/')==0 ]
polCorpus = [ [startToken] + [w.lower() for w in myCorpus.words(f)] + [endToken] for f in fileNames if f.find('J-Politics'+'/')==0 ]
culCorpus = [ [startToken] + [w.lower() for w in myCorpus.words(f)] + [endToken] for f in fileNames if f.find('C-Culture'+'/')==0 ]
socCorpus = [ [startToken] + [w.lower() for w in myCorpus.words(f)] + [endToken] for f in fileNames if f.find('D-Society'+'/')==0 ]
zCorpus = [ [startToken] + [w.lower() for w in myCorpus.words(f)] + [endToken] for f in fileNames if f.find('Z'+'/')==0 ]

testEcoCorpus, trainEcoCorpus = splitSentCorpus(ecoCorpus)
testMilCorpus, trainMilCorpus = splitSentCorpus(milCorpus)
testPolCorpus, trainPolCorpus = splitSentCorpus(polCorpus)
testCulCorpus, trainCulCorpus = splitSentCorpus(culCorpus)


#############################################################
#######   Тестване на Бейсов класификатор
#############################################################

condProbM, priorM, VM = trainMultinomialNB([trainEcoCorpus,trainMilCorpus,trainPolCorpus,trainCulCorpus])

gamma = lambda text : applyMultinomialNB(priorM, condProbM, text)
testClassifier([testEcoCorpus,testMilCorpus,testPolCorpus,testCulCorpus], gamma)

condProbM, priorM, VM = trainMultinomialNB([trainEcoCorpus,trainMilCorpus])

gamma = lambda text : applyMultinomialNB(priorM, condProbM, text)
testClassifier([testEcoCorpus,testMilCorpus], gamma)


#############################################################
###  Влагане на думите
#############################################################

C, words, word2ind = co_occurrence_matrix(ecoCorpus+milCorpus+polCorpus+culCorpus+socCorpus+zCorpus)
X = PMI_matrix(C)
X_reduced = SVD_k_dim(X)

X_lengths_100d = np.linalg.norm(X_reduced, axis=1)
X_normalized_100d = X_reduced / X_lengths_100d[:, np.newaxis]

#############################################################

trainX = np.concatenate((
                         corpusEmbedding(trainEcoCorpus,X_normalized_100d,word2ind),
                         corpusEmbedding(trainMilCorpus,X_normalized_100d,word2ind)
                         ))
trainY = np.concatenate((
                         np.ones(len(trainEcoCorpus),dtype='int32')*0,
                         np.ones(len(trainMilCorpus),dtype='int32')*1
                         ))
testX = np.concatenate((
                        corpusEmbedding(testEcoCorpus,X_normalized_100d,word2ind),
                        corpusEmbedding(testMilCorpus,X_normalized_100d,word2ind)
                        ))
testY = np.concatenate((
                        np.ones(len(testEcoCorpus),dtype='int32')*0,
                        np.ones(len(testMilCorpus),dtype='int32')*1
                        ))

w0 = np.random.normal(0.,1.,100)
b0 = np.random.normal(0., 1., 1)

w,b,ceList,tceList = gradientDescend(trainX,trainY,testX,testY,w0,b0,crossEntropyS,gradCrossEntropyS,alpha=1.)

plt.plot([*range(len(ceList))],ceList)
plt.plot([*range(len(tceList))],tceList)
plt.show()

gamma = lambda text : 1 if sigmoid(np.dot(w,docVector(text, X_normalized_100d, word2ind))+b)>0.5 else 0

testClassifier([testEcoCorpus,testMilCorpus], gamma)


#############################################################

trainX = np.concatenate((
                         corpusEmbedding(trainEcoCorpus,X_normalized_100d,word2ind),
                         corpusEmbedding(trainMilCorpus,X_normalized_100d,word2ind),
                         corpusEmbedding(trainPolCorpus,X_normalized_100d,word2ind),
                         corpusEmbedding(trainCulCorpus,X_normalized_100d,word2ind)
                         ))
trainY = np.concatenate((
                         np.ones(len(trainEcoCorpus),dtype='int32')*0,
                         np.ones(len(trainMilCorpus),dtype='int32')*1,
                         np.ones(len(trainPolCorpus),dtype='int32')*2,
                         np.ones(len(trainCulCorpus),dtype='int32')*3
                         ))

testX = np.concatenate((
                         corpusEmbedding(testEcoCorpus,X_normalized_100d,word2ind),
                         corpusEmbedding(testMilCorpus,X_normalized_100d,word2ind),
                         corpusEmbedding(testPolCorpus,X_normalized_100d,word2ind),
                         corpusEmbedding(testCulCorpus,X_normalized_100d,word2ind)
                         ))
testY = np.concatenate((
                         np.ones(len(testEcoCorpus),dtype='int32')*0,
                         np.ones(len(testMilCorpus),dtype='int32')*1,
                         np.ones(len(testPolCorpus),dtype='int32')*2,
                         np.ones(len(testCulCorpus),dtype='int32')*3
                         ))

W0 = np.random.normal(0.,1.,size=(100,4))
b0 = np.random.normal(0., 1., 4)

W,b,ceList,tceList = gradientDescend(trainX,trainY,testX,testY,W0,b0,crossEntropyM,gradCrossEntropyM,alpha=1.)

plt.plot([*range(len(ceList))],ceList)
plt.plot([*range(len(tceList))],tceList)
plt.show()

gamma = lambda text : np.argmax(softmaxV(np.dot(docVector(text, X_normalized_100d, word2ind),W)+b))
testClassifier([testEcoCorpus,testMilCorpus,testPolCorpus,testCulCorpus], gamma)

