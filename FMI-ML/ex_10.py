#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Упражнение 10
###
#############################################################################

import nltk
from nltk.corpus import PlaintextCorpusReader
import sys
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pprint

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
###  Разбиване на корпус на тестов и тренинг и извличане на речник
#############################################################
def splitSentCorpus(fullSentCorpus, testFraction = 0.1):
    random.seed(42)
    random.shuffle(fullSentCorpus)
    testCount = int(len(fullSentCorpus) * testFraction)
    testSentCorpus = fullSentCorpus[:testCount]
    trainSentCorpus = fullSentCorpus[testCount:]
    return testSentCorpus, trainSentCorpus

startToken = '<START>'
endToken = '<END>'
unkToken = '<UNK>'


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
    words = [ w for w,_ in L[:limit] ] + [unkToken]
    word2ind = { w:i for i,w in enumerate(words)}
    pb.stop()
    return words, word2ind

################################################################################
################################################################################
####
#### Обектна имплементация на Backpropagation с Numpy операции
####
################################################################################
################################################################################

class compNode:
    def __init__(self, predecessors, trainable = True):
        self.predecessors = predecessors
        self.trainable = trainable
        self.value = None
        self.grad = None
    
    def calcValue(self): ## трябва да се дефинира за конкретния връх като се извика setValue
        return
    
    def propagateGrad(self, grad):
        if not self.grad:
            self.grad = grad
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
    def __init__(self, topNode):
        self.topNode = topNode
        self.sortedNodes = getSortedNodes(topNode,[])
        self.paramNodes = [ v for v in self.sortedNodes if v.trainable and not v.predecessors ]
        self.dataNodes = [ v for v in self.sortedNodes if not v.trainable and not v.predecessors ]
    
    def setParameters(self, params):
        for i, p in enumerate(params):
            self.paramNodes[i].value = p

    def setData(self, data):
        for i, d in enumerate(data):
            self.dataNodes[i].value = d

    def saveModelParams(self, fileName):
        with open(fileName, 'wb') as f:
            for p in self.paramNodes:
                np.save(f, p.value)

    def loadModelParams(self, fileName):
        with open(fileName, 'rb') as f:
            for p in self.paramNodes:
                p.value = np.load(f, allow_pickle=True)

    def forward(self):
        for v in self.sortedNodes:
            v.calcValue()

    def backwards(self):
        for v in self.sortedNodes:
            v.grad = None
        self.topNode.propagateGrad(1)
        for v in reversed(self.sortedNodes):
            v.propagateBack()
                
    def updateModel(self,alpha):
        for p in self.paramNodes:
            p.value -= alpha * p.grad
    
    def calcLoss(self,testData,batchSize):
        loss = 0.
        samples = len(testData[0])
        for i in range(0,samples,batchSize):
            li = min(i+batchSize, samples)
            batchData = [d[i:li] for d in testData ]
            self.setData(batchData)
            self.forward()
            loss += (li-i) * self.topNode.value
        return loss / samples
    
    def batchedStochasticGradient(self, trainData, testData, batchSize, alpha = 1., maxEpoch = 100000, printInterval = 100, saveInterval = 10000, fileToSave = None):
        ceList = []
        tceList = []
        epoch = 0
        batch = 0
        samples = np.arange(len(trainData[0]), dtype='int32')
        batchesInEpoch = len(samples) // batchSize
        while epoch<maxEpoch:
            np.random.shuffle(samples)
            for i in range(0,len(samples),batchSize):
                if fileToSave and batch > 0 and batch % saveInterval == 0:
                    self.saveModelParams(fileToSave)
                if batch % printInterval == 0:
                    ce = self.topNode.value
                    tce = self.calcLoss(testData,batchSize)
                    ceList.append(ce)
                    tceList.append(tce)
                    print('Epoch: '+str(epoch) + ', Batch: '+str(batch % batchesInEpoch)+'/'+str(batchesInEpoch)+', Train loss: '+str(ce)+', Test loss: '+str(tce))
                idx = samples[i:min(i+batchSize, len(samples))]
                batchData = [d[idx] for d in trainData ]
                self.setData(batchData)
                self.forward()
                self.backwards()
                self.updateModel(alpha)
                batch += 1
            epoch += 1
        return ceList, tceList

################################################################################

class termFrequencyNode(compNode):
    def setDictionarySize(self,size):
        self.dictionarySize = size
    def calcValue(self):
        ### c следва да бъде списък от контексти с индекси
        c = self.predecessors[0].value
        S = len(c)
        self.value = np.zeros((S,self.dictionarySize), dtype = 'int32')
        for i in range(S):
            for j in c[i]:
                self.value[i,j] += 1

class mulMatrixMatrixNode(compNode):
    def calcValue(self):
        x = self.predecessors[0].value
        y = self.predecessors[1].value
        self.value = np.dot(x,y)
    def derivativeMult(self,i):
        if i == 0:
            y = self.predecessors[1].value
            return np.dot(self.grad,y.transpose())
        else:
            x = self.predecessors[0].value
            return np.dot(x.transpose(),self.grad)

def softmaxM(U):
    ### U следва да бъде матрица с размерност: (S,M)
    U = np.exp(U)
    tmp = np.sum(U,axis=1)[:,np.newaxis]
    U /= tmp
    return U

class crossEntropySoftmaxNode(compNode):
    def calcValue(self):
        ### t следва да бъде матрица с размерност: (S,M)
        ### w следва да бъде вектор с размерност S от индекси на думи
        t = self.predecessors[0].value
        w = self.predecessors[1].value
        self.S = t.shape[0]
        self.v = softmaxM(t)
        p = self.v[np.arange(self.S, dtype='int32'),w]
        self.value = -np.mean(np.log(p))
    def derivativeMult(self,i):
        w = self.predecessors[1].value
        d = -self.v
        d[np.arange(self.S, dtype='int32'),w] += 1.
        return self.grad * (-d/self.S)

#############################################################

def extractContexts(corpus, window_size, words, word2ind):
    pb = progressBar()
    pb.start(len(corpus))
    unk = word2ind[unkToken]

    centers = []
    contexts = []
    for doc in corpus:
        pb.tick()
        for wi in range(len(doc)):
            i = word2ind.get(doc[wi], unk)
            context = []
            for k in range(1,window_size+1):
                if wi-k>=0:
                    j = word2ind.get(doc[wi-k], unk)
                    context.append(j)
                if wi+k<len(doc) and doc[wi+k] in word2ind:
                    j = word2ind.get(doc[wi+k], unk)
                    context.append(j)
            if len(context)==0: continue
            centers.append(i)
            contexts.append(context)
    pb.stop()
    return np.array(centers, dtype = 'int32'),np.array(contexts, dtype=object)

#############################################################
#######   Зареждане на корпуса
#############################################################

corpus_root = 'JOURNALISM.BG/C-MassMedia'
myCorpus = PlaintextCorpusReader(corpus_root, '.*\.txt')

corpus = [ [startToken] + [w.lower() for w in sent] + [endToken] for sent in myCorpus.sents()]

words, word2ind = extractDictionary(corpus)

testCorpus, trainCorpus = splitSentCorpus(corpus)

window_size = 4
trainW, trainC = extractContexts(trainCorpus, window_size, words, word2ind)
testW, testC = extractContexts(testCorpus, window_size, words, word2ind)


w = compNode(None,trainable=False)
C = compNode(None,trainable=False)
U = compNode(None)
V = compNode(None)
chi = termFrequencyNode([C],trainable=False)
chi.setDictionarySize(len(words))
VC = mulMatrixMatrixNode([chi,V])
Z = mulMatrixMatrixNode([VC,U])
h = crossEntropySoftmaxNode([Z,w])

embDim = 50
U0 = (np.random.rand(embDim, len(words)) - 0.5) / embDim
V0 = (np.random.rand(len(words), embDim) - 0.5) / embDim

word2vec = model(h)
word2vec.setParameters([np.copy(V0),np.copy(U0)])

cefList, tcefList = word2vec.batchedStochasticGradient([trainC,trainW], [testC[:1000],testW[:1000]], 100, maxEpoch = 1, printInterval = 10, saveInterval = 100, fileToSave = 'test.save')

# Ако имаме вече натрениран модел може директно да го заредим
#   word2vec.loadModelParams('w2v.params')
#
E = np.concatenate([U.value.transpose(),V.value],axis=1)

def SVD_k_dim(X, k=100, n_iters = 10):
    # Документация на метода има на https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
    print("Running Truncated SVD over %i words..." % (X.shape[0]))
    svd = TruncatedSVD(n_components=k, n_iter=n_iters)
    svd.fit(X)
    X_reduced = svd.transform(X)
    print("Done.")
    return X_reduced
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


def plot_embeddings_pairs_3d(M, word2ind, wordPairs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xs = M[:,0]
    ys = M[:,1]
    zs = M[:,2]
    for u,v in wordPairs:
        i=word2ind[u]
        j=word2ind[v]
        ax.plot(xs[[i,j]], ys[[i,j]], zs[[i,j]], color= 'red')
        ax.text(xs[i]+0.001, ys[i]+0.001, zs[i]+0.001, u)
        ax.text(xs[j]+0.001, ys[j]+0.001, zs[j]+0.001, v)
    plt.show()


E_reduced =SVD_k_dim(E,k=3)
E_normalized_3d = E_reduced /np.linalg.norm(E_reduced, axis=1)[:, np.newaxis]
sampleWords = ['кола', 'автомобил', 'румъния', 'министър', 'президент', 'гърция', 'футбол', 'спорт', 'баскетбол', 'българия', 'театър', 'кино', 'опера']

plot_embeddings_3d(E_normalized_3d, word2ind, sampleWords)

sampleWordPairs = [('румъния', 'букурещ'), ('италия', 'рим'), ('франция', 'париж'), ('унгария', 'будапеща'), ('българия', 'софия'), ('германия', 'берлин')]

plot_embeddings_pairs_3d(E_normalized_3d, word2ind, sampleWordPairs)

#############################################################
E_normalized = E /np.linalg.norm(E, axis=1)[:, np.newaxis]

def most_similar(w,word2ind,words,C,limit=10):
    i = word2ind[w]
    L = np.dot(C,C[i]).tolist()
    L = sorted([(words[i],s) for i,s in enumerate(L)], key = lambda x: x[1] , reverse=True)
    return L[:limit]

pprint.pprint(most_similar('гърция',word2ind,words,E_normalized,limit=5))
pprint.pprint(most_similar('футбол',word2ind,words,E_normalized,limit=5))
pprint.pprint(most_similar('град',word2ind,words,E_normalized,limit=5))
pprint.pprint(most_similar('съд',word2ind,words,E_normalized,limit=5))

#############################################################

print('Перплексията на Word2Vec CBOW модела е: ', np.exp(word2vec.calcLoss([testC[:100000],testW[:100000]],100)))

#############################################################


