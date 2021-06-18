#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Упражнение 11
###
#############################################################################

import sys
import nltk
from nltk.corpus import PlaintextCorpusReader
import numpy as np
import torch


X = torch.tensor([[1.2,2,3],[4,5,6]])
Y = torch.tensor([[3,2,1],[2,3,4.1]], requires_grad=True)
A = torch.rand(3,4, requires_grad=True)
B = torch.matmul(X+Y,A)
C = torch.sum(-2 * B)
C.backward()

print(A.grad)
print(Y.grad)

S = torch.mean(A)
S.backward()
print(A.grad)
print(Y.grad)

A.grad = None
Y.grad = None

S = torch.mean(torch.matmul(Y,A))
S.backward()

print(A.grad)
print(Y.grad)

with torch.no_grad():
    A -= 1.2 * A.grad
    Y -= 1.2 * Y.grad

print(Y)

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

def extractData(corpus, order, word2ind):
    pb = progressBar()
    pb.start(len(corpus))
    unk = word2ind[unkToken]
    start = word2ind[startToken]

    points = sum(len(s)-1 for s in corpus)
    
    target = np.empty(points, dtype='int32')
    context = np.empty((points,order-1), dtype='int32')
    p = 0
    for doc in corpus:
        pb.tick()
        for wi in range(1,len(doc)):
            i = word2ind.get(doc[wi], unk)
            target[p] = i
            sample = []
            for k in range(1,order):
                if wi-k < 0:
                    j = start
                else:
                    j = word2ind.get(doc[wi-k], unk)
                context[p,k-1] = j
            p += 1
    pb.stop()
    return target, context

#############################################################
#######   Зареждане на корпуса
#############################################################

corpus_root = 'JOURNALISM.BG/C-MassMedia'
myCorpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
startToken = '<START>'
endToken = '<END>'
unkToken = '<UNK>'

corpus = [ [startToken] + [w.lower() for w in sent] + [endToken] for sent in myCorpus.sents()]

words, word2ind = extractDictionary(corpus)

order = 4
target, context = extractData(corpus, order, word2ind)

emb_size = 50
hid_size = 100

L = len(words)

batchSize = 1000
idx = np.arange(len(target), dtype='int32')
np.random.shuffle(idx)
learning_rate = 1.

#############################################################
#######   Първи вариант
#############################################################


E = torch.empty(L, emb_size, requires_grad = True)
W1 = torch.empty((order-1)*emb_size, hid_size, requires_grad = True)
b1 = torch.empty(hid_size, requires_grad = True)
W2 = torch.empty(hid_size, L, requires_grad = True)
b2 = torch.empty(L, requires_grad = True)

torch.nn.init.normal_(E)
torch.nn.init.normal_(W1)
torch.nn.init.normal_(b1)
torch.nn.init.normal_(W2)
torch.nn.init.normal_(b2)

sigmoid_fn = torch.nn.Sigmoid()

for b in range(0,len(idx),batchSize):

    batchIdx = idx[b:min(b+batchSize,len(idx))]
    S = len(batchIdx)
    batchTarget = torch.tensor(target[batchIdx], dtype=torch.long)
    batchContext = context[batchIdx]

    X = E[batchContext].view(S,(order-1) * emb_size)
    h = sigmoid_fn(torch.matmul(X,W1) + b1)
    z = torch.matmul(h,W2) + b2
    
    t = torch.exp(z)
    s = torch.sum(t,axis=1)
    z = t/s.unsqueeze(1)
    p = z[torch.arange(S),batchTarget]
    H = -torch.mean(torch.log(p))

    H.backward()

    with torch.no_grad():
        E -= learning_rate * E.grad
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad
        # Manually zero the gradients
        E.grad = None
        W1.grad = None
        b1.grad = None
        W2.grad = None
        b2.grad = None

    if b % 10000 == 0:
        print(b, '/', len(idx), H.item())


#############################################################
#######   Втори вариант
#############################################################

torch.nn.init.normal_(E)
torch.nn.init.normal_(W1)
torch.nn.init.normal_(b1)
torch.nn.init.normal_(W2)
torch.nn.init.normal_(b2)

loss_fn = torch.nn.CrossEntropyLoss()

for b in range(0,len(idx),batchSize):
    
    batchIdx = idx[b:min(b+batchSize,len(idx))]
    S = len(batchIdx)
    batchTarget = torch.tensor(target[batchIdx], dtype=torch.long)
    batchContext = context[batchIdx]
    
    X = E[batchContext].view(S,(order-1) * emb_size)
    h = sigmoid_fn(torch.matmul(X,W1) + b1)
    z = torch.matmul(h,W2) + b2
    H = loss_fn(z,batchTarget)
    
    H.backward()
    
    with torch.no_grad():
        E -= learning_rate * E.grad
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad
        # Manually zero the gradients
        E.grad = None
        W1.grad = None
        b1.grad = None
        W2.grad = None
        b2.grad = None
    
    if b % 10000 == 0:
        print(b, '/', len(idx), H.item())


#############################################################
#######   Трети вариант
#############################################################

device = torch.device("cuda:0")

E = torch.empty(L, emb_size, requires_grad = True, device = device)
W1 = torch.empty((order-1)*emb_size, hid_size, requires_grad = True, device = device)
b1 = torch.empty(hid_size, requires_grad = True, device = device)
W2 = torch.empty(hid_size, L, requires_grad = True, device = device)
b2 = torch.empty(L, requires_grad = True, device = device)

torch.nn.init.normal_(E)
torch.nn.init.normal_(W1)
torch.nn.init.normal_(b1)
torch.nn.init.normal_(W2)
torch.nn.init.normal_(b2)


for b in range(0,len(idx),batchSize):
    
    batchIdx = idx[b:min(b+batchSize,len(idx))]
    S = len(batchIdx)
    batchTarget = torch.tensor(target[batchIdx], dtype=torch.long, device = device)
    batchContext = context[batchIdx]
    
    X = E[batchContext].view(S,(order-1) * emb_size)
    h = sigmoid_fn(torch.matmul(X,W1) + b1)
    z = torch.matmul(h,W2) + b2
    H = loss_fn(z,batchTarget)
    
    H.backward()
    
    with torch.no_grad():
        E -= learning_rate * E.grad
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad
        # Manually zero the gradients
        E.grad = None
        W1.grad = None
        b1.grad = None
        W2.grad = None
        b2.grad = None
    
    if b % 10000 == 0:
        print(b, '/', len(idx), H.item())


