#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Упражнение 9
###
#############################################################################

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
####    Логистична регресия -- Бинарен класификатор от упр. 8
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
####    спускане по градиента от упр. 8
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

w,b,ceList,tceList = gradientDescend(trainX,trainY,testX,testY,np.copy(w0),np.copy(b0),crossEntropyS,gradCrossEntropyS,alpha=1.)

plt.plot([*range(len(ceList))],ceList)
plt.plot([*range(len(tceList))],tceList)
plt.show()

################################################################################
################################################################################
####
#### Обектна имплементация на Backpropagation с Numpy операции
####
################################################################################
################################################################################

################################################################################
#### Първи вариант
################################################################################

class compNode:
    ################################################################################
    #### Базов обект за връх в изчислителния граф -- първи вариант
    ################################################################################

    def __init__(self, predecessors, trainable = True):
        self.predecessors = predecessors
        self.zeroGrad = True
        self.trainable = trainable

    def getValue(self):
        return self.value

    def setValue(self,value):
        self.value = value
        self.zeroGrad = True

    def calcValue(self): ## трябва да се дефинира за конкретния връх като се извика setValue
        return

    def propagateGrad(self, grad):
        if self.zeroGrad:
            self.grad = grad
            self.zeroGrad = False
        else:
            self.grad += grad

    def derivative(self,i): ## трябва да се дефинира за конкретния връх
        return
    
    def propagateBack(self):
        if not self.predecessors: return
        for i,p in enumerate(self.predecessors):
            if p.trainable:
                partialGrad = np.dot(self.grad,self.derivative(i))
                p.propagateGrad(partialGrad)

################################################################################
#### Конкретни инстанции на обекти за върхове в изчислителния граф -- първи вариант
################################################################################

class logNode(compNode):
    def calcValue(self):
        x = self.predecessors[0].getValue()
        self.setValue(np.log(x))
    def derivative(self,i):
        x = self.predecessors[0].getValue()
        return np.diag(1/x)

class sigmoidNode(compNode):
    def calcValue(self):
        x = self.predecessors[0].getValue()
        self.setValue(sigmoid(x))
    def derivative(self,i):
        argument = self.predecessors[0].getValue()
        s = sigmoid(argument)
        return np.diag(s*(1-s))

class minusMeanNode(compNode):
    def calcValue(self):
        x = self.predecessors[0].getValue()
        self.setValue(-np.mean(x))
    def derivative(self,i):
        x = self.predecessors[0].getValue()
        S=x.shape[0]
        return -1/S * np.ones(S)

class probNode(compNode):
    def calcValue(self):
        v = self.predecessors[0].getValue()
        y = self.predecessors[1].getValue()
        self.setValue((1-y) + (2*y-1)*v)
    def derivative(self,i):
        assert i==0
        y = self.predecessors[1].getValue()
        return np.diag(2*y-1)

class plusNode(compNode):
    def calcValue(self):
        x = self.predecessors[0].getValue()
        y = self.predecessors[1].getValue()
        self.setValue(x+y)
    def derivative(self,i):
        S = self.value.shape[0]
        return np.eye(S)

class mulNode(compNode):
    def calcValue(self):
        x = self.predecessors[0].getValue()
        y = self.predecessors[1].getValue()
        self.setValue(np.dot(x,y))
    def derivative(self,i):
        j = 1 - i
        u = self.predecessors[j].getValue()
        return u

class copyNode(compNode):
    def calcValue(self):
        self.setValue(self.predecessors[0].getValue())
    def derivative(self,i):
        S = self.grad.shape
        return np.ones(S)

class constNode(compNode):
    def calcValue(self):
        self.setValue(self.value)

################################################################################
#### Създаване на изчислителен граф за логистичната регресия -- първи вариант
################################################################################

x = constNode(None,trainable=False)
y = constNode(None,trainable=False)
w = constNode(None)
b = constNode(None)
u = mulNode([x,w])
bS = copyNode([b])
t = plusNode([u,bS])
v = sigmoidNode([t])
p = probNode([v,y])
l = logNode([p])
h = minusMeanNode([l])


################################################################################
#### Топологично сортиране на върховете на изчислителен граф
################################################################################
def getSortedNodes(t,L):
    if t in L: return L
    if t.predecessors:
        for p in t.predecessors:
            L = getSortedNodes(p,L)
    L.append(t)
    return L


################################################################################
#### Базов обект за модел на невронна мрежа
#### Съдържа имплементация на Backpropagation
#### и стохастично спускане по градиента
################################################################################

class model:
    def __init__(self, topNode, paramNodes, dataNodes):
        self.topNode = topNode
        self.paramNodes = paramNodes
        self.dataNodes = dataNodes
        self.sortedNodes = getSortedNodes(topNode,[])

    def setParameters(self, params):
        for i, p in enumerate(params):
            self.paramNodes[i].value = p

    def setData(self, data):
        for i, d in enumerate(data):
            self.dataNodes[i].value = d
    
    def forward(self):
        for n in self.sortedNodes:
            n.calcValue()

    def backwards(self):
        self.topNode.propagateGrad(1)
        for n in reversed(self.sortedNodes):
            n.propagateBack()

    def updateModel(self,alpha):
        for p in self.paramNodes:
            p.value -= alpha * p.grad

    def calcLoss(self,testData):
        self.setData(testData)
        self.forward()
        return self.topNode.value

    def batchedStochasticGradient(self, initialParams, trainData, testData, batchSize, alpha = 1., maxEpoch = 100000, printInterval = 1000):
        self.setParameters(initialParams)
        ceList = []
        tceList = []
        epoch = 0
        samples = np.arange(trainData[0].shape[0], dtype='int32')
        while epoch<maxEpoch:
            if epoch % printInterval == 0:
                tce = self.calcLoss(testData)
                ce = self.calcLoss(trainData)
                print(epoch, ce, tce)
                ceList.append(ce)
                tceList.append(tce)
            np.random.shuffle(samples)
            for i in range(0,len(samples),batchSize):
                idx = samples[i:min(i+batchSize, len(samples))]
                batchData = [d[idx] for d in trainData ]
                self.setData(batchData)
                self.forward()
                self.backwards()
                self.updateModel(alpha)
            epoch += 1
        return ceList, tceList

################################################################################
#### Създаване на конкретен модел за логистичната регресия
################################################################################
logistic = model(h,[w,b],[x,y])

################################################################################
#### Трениране на модела с пълно спускане по градиента -- бавно!
################################################################################

logistic.batchedStochasticGradient([np.copy(w0),np.copy(b0)], [trainX,trainY], [testX,testY], trainX.shape[0], maxEpoch = 2001, printInterval = 1000)

################################################################################
################################################################################
####
#### Обектна имплементация на Backpropagation с Numpy операции
####
################################################################################
################################################################################
################################################################################
#### Втори подобрен вариант
################################################################################

class compNode:
    def __init__(self, predecessors, trainable = True):
        self.predecessors = predecessors
        self.zeroGrad = True
        self.trainable = trainable
    
    def getValue(self):
        return self.value
    
    def setValue(self,value):
        self.value = value
        self.zeroGrad = True
    
    def calcValue(self): ## трябва да се дефинира за конкретния връх като се извика setValue
        return
    
    def propagateGrad(self, grad):
        if self.zeroGrad:
            self.grad = grad
            self.zeroGrad = False
        else:
            self.grad += grad

    def derivativeMult(self,i): ## трябва да се дефинира за конкретния връх
        return
    
    def propagateBack(self):
        if not self.predecessors: return
        for i,p in enumerate(self.predecessors):
            if p.trainable:
                partialGrad = self.derivativeMult(i)
                p.propagateGrad(partialGrad)

################################################################################
#### Конкретни инстанции на обекти за върхове в изчислителния граф -- втори вариант
################################################################################

################################################################################
#### Тук дефинираме нов, обединяващ връх за кросентропия
################################################################################
class crossEntropyNode(compNode):
    def calcValue(self):
        t = self.predecessors[0].getValue()
        y = self.predecessors[1].getValue()
        self.v = sigmoid(t)
        p = (1-y) + (2*y-1) * self.v
        self.setValue(-np.mean(np.log(p)))
    def derivativeMult(self,i):
        y = self.predecessors[1].getValue()
        S = y.shape[0]
        return self.grad * (-(y-self.v)/S)

class plusNode(compNode):
    def calcValue(self):
        x = self.predecessors[0].getValue()
        y = self.predecessors[1].getValue()
        self.setValue(x+y)
    def derivativeMult(self,i):
        return self.grad

class mulNode(compNode):
    def calcValue(self):
        x = self.predecessors[0].getValue()
        y = self.predecessors[1].getValue()
        self.setValue(np.dot(x,y))
    def derivativeMult(self,i):
        j = 1 - i
        u = self.predecessors[j].getValue()
        return np.dot(self.grad,u)

class copyNode(compNode):
    def calcValue(self):
        self.setValue(self.predecessors[0].getValue())
    def derivativeMult(self,i):
        return np.sum(self.grad)

class constNode(compNode):
    def calcValue(self):
        self.setValue(self.value)

################################################################################
#### Създаване на изчислителен граф и модел за логистичната регресия -- втори вариант
################################################################################

x = constNode(None,trainable=False)
y = constNode(None,trainable=False)
w = constNode(None)
b = constNode(None)
u = mulNode([x,w])
bS = copyNode([b])
t = plusNode([u,bS])
h = crossEntropyNode([t,y])

logistic = model(h,[w,b],[x,y])

################################################################################
#### Трениране на модела с пълно спускане и партидно стохастично спускане
################################################################################

cefList, tcefList = logistic.batchedStochasticGradient([np.copy(w0),np.copy(b0)], [trainX,trainY], [testX,testY], trainX.shape[0], maxEpoch = 100000, printInterval = 1000)

cebList, tcebList = logistic.batchedStochasticGradient([np.copy(w0),np.copy(b0)], [trainX,trainY], [testX,testY], 100, maxEpoch = 20000, printInterval = 1000)


################################################################################
#### визуализация
################################################################################

plt.plot([*range(len(cefList))],cefList)
plt.plot([*range(len(tcefList))],tcefList)
plt.plot([*range(len(cebList))],cebList)
plt.plot([*range(len(tcebList))],tcebList)
plt.legend(['Full gradient train','Full gradient test','Stochastic gradient train','Stochastic gradient test'])
plt.show()

################################################################################
#### Трениране на модела със стандартно стохастично спускане
################################################################################

cesList, tcesList = logistic.batchedStochasticGradient([np.copy(w0),np.copy(b0)], [trainX,trainY], [testX,testY], 1, maxEpoch = 200, printInterval = 10)

plt.plot([*range(len(cebList))],cebList)
plt.plot([*range(len(tcebList))],tcebList)
plt.plot([*range(len(cesList))],cesList)
plt.plot([*range(len(tcesList))],tcesList)
plt.legend(['Batched Stochastic gradient train','Batched Stochastic gradient test','Standard Stochastic gradient train','Standard Stochastic gradient test'])
plt.show()
