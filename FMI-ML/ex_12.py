#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Упражнение 12
###
#############################################################################

import sys
import nltk
from nltk.corpus import PlaintextCorpusReader
import numpy as np
import torch
import random
import math

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

def splitSentCorpus(fullSentCorpus, testFraction = 0.1):
    random.seed(42)
    random.shuffle(fullSentCorpus)
    testCount = int(len(fullSentCorpus) * testFraction)
    testSentCorpus = fullSentCorpus[:testCount]
    trainSentCorpus = fullSentCorpus[testCount:]
    return testSentCorpus, trainSentCorpus

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

testCorpus, trainCorpus  = splitSentCorpus(corpus, testFraction = 0.01)

order = 4
target, context = extractData(trainCorpus, order, word2ind)

emb_size = 50
hid_size = 100

L = len(words)

batchSize = 1000
idx = np.arange(len(target), dtype='int32')
np.random.shuffle(idx)
learning_rate = 1.

#device = torch.device("cpu")
#device = torch.device("cuda:0")
device = torch.device("cuda:1")

#############################################################
#######   Параметри на модела
#############################################################

E = torch.empty(L, emb_size, requires_grad = True)
W1 = torch.empty((order-1)*emb_size, hid_size, requires_grad = True)
b1 = torch.empty(hid_size, requires_grad = True)
W2 = torch.empty(hid_size, L, requires_grad = True)
b2 = torch.empty(L, requires_grad = True)

#############################################################
#######   Стар вариант
#############################################################

torch.nn.init.normal_(E)
torch.nn.init.normal_(W1)
torch.nn.init.normal_(b1)
torch.nn.init.normal_(W2)
torch.nn.init.normal_(b2)

for b in range(0,len(idx),batchSize):
    
    batchIdx = idx[b:min(b+batchSize,len(idx))]
    S = len(batchIdx)
    batchTarget = torch.tensor(target[batchIdx], dtype=torch.long)
    batchContext = context[batchIdx]
    
    X = E[batchContext].view(S,(order-1) * emb_size)
    h = torch.sigmoid(torch.matmul(X,W1) + b1)
    z = torch.matmul(h,W2) + b2
    H = torch.nn.functional.cross_entropy(z,batchTarget)
    
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
#######   Дефиниране на нова функция за афинна трансформация
#############################################################

class AfineFunction(torch.autograd.Function):
    
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # This function implements output = input @ weight + bias
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        output = torch.mm(input,weight)
        output += bias.unsqueeze(0)
        return output
    
    # This function gets the gradient for its output
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        
        grad_input = torch.mm(grad_output, weight.t())
        grad_weight = torch.mm(input.t(), grad_output)
        grad_bias = grad_output.sum(0)
        
        return grad_input, grad_weight, grad_bias

#############################################################
#######    вариант използващ афинната функция
#############################################################

torch.nn.init.normal_(E)
torch.nn.init.normal_(W1)
torch.nn.init.normal_(b1)
torch.nn.init.normal_(W2)
torch.nn.init.normal_(b2)

for b in range(0,len(idx),batchSize):
    
    batchIdx = idx[b:min(b+batchSize,len(idx))]
    batchTarget = torch.tensor(target[batchIdx], dtype=torch.long)
    batchContext = context[batchIdx]
    
    X = E[batchContext].flatten(1,2)
    h = torch.sigmoid(AfineFunction.apply(X,W1,b1))
    z = AfineFunction.apply(h,W2,b2)
    H = torch.nn.functional.cross_entropy(z,batchTarget)
    
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
#######   Вариант с използване на модул
#############################################################

class LModel(torch.nn.Module):
    def __init__(self, L, emb_size, hid_size, order):
        super(LModel, self).__init__()
        
        self.E = torch.nn.Parameter(torch.rand(L, emb_size)-0.5)
        self.W1 = torch.nn.Parameter(torch.rand((order-1)*emb_size, hid_size)-0.5)
        self.b1 = torch.nn.Parameter(torch.rand(hid_size)-0.5)
        self.W2 = torch.nn.Parameter(torch.rand(hid_size, L)-0.5)
        self.b2 = torch.nn.Parameter(torch.rand(L)-0.5)
    
    def forward(self, context, target):
        device = next(model.parameters()).device
        contextTensor = torch.tensor(context, dtype=torch.long, device = device)
        targetTensor = torch.tensor(target, dtype=torch.long, device = device)
        
        X = self.E[contextTensor].flatten(1,2)
        h = torch.sigmoid(torch.matmul(X,self.W1) + self.b1)
        z = torch.matmul(h,self.W2) + self.b2
        H = torch.nn.functional.cross_entropy(z,targetTensor)
        return H

model = LModel(L, emb_size, hid_size, order).to(device)
for p in model.parameters(): print(p.shape)

for b in range(0,len(idx),batchSize):
    
    batchIdx = idx[b:min(b+batchSize,len(idx))]
    H = model(context[batchIdx],target[batchIdx])
    
    model.zero_grad()
    H.backward()
    
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

    if b % 10000 == 0:
        print(b, '/', len(idx), H.item())

#############################################################
#######   Вариант с модул, в който са вложени други модули
#############################################################

class LModel(torch.nn.Module):
    def __init__(self, L, emb_size, hid_size, order):
        super(LModel, self).__init__()
        
        self.embedding = torch.nn.Embedding(L, emb_size)
        self.layer1 = torch.nn.Linear((order-1) * emb_size, hid_size)
        self.layer2 = torch.nn.Linear(hid_size, L)
    
    def forward(self, context, target):
        device = next(model.parameters()).device
        targetTensor = torch.tensor(target, dtype=torch.long, device = device)
        contextTensor = torch.tensor(context, dtype=torch.long, device = device)

        X = self.embedding(contextTensor).flatten(1,2)
        h = torch.sigmoid(self.layer1(X))
        z = self.layer2(h)
        return torch.nn.functional.cross_entropy(z,targetTensor)

model = LModel(L, emb_size, hid_size, order).to(device)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.000001)

for b in range(0,len(idx),batchSize):
    
    batchIdx = idx[b:min(b+batchSize,len(idx))]
    H = model(context[batchIdx],target[batchIdx])
    
    optimizer.zero_grad()
    H.backward()
    optimizer.step()

    if b % 10000 == 0:
        print(b, '/', len(idx), H.item())

#############################################################
#######   Вариант с използване на последователност от модули
#############################################################

model = torch.nn.Sequential(
                            torch.nn.Embedding(L, emb_size),
                            torch.nn.Flatten(1,2),
                            torch.nn.Linear((order-1) * emb_size, hid_size),
                            torch.nn.Sigmoid(),
                            torch.nn.Linear(hid_size, L)
                            ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.000001)

for b in range(0,len(idx),batchSize):
    
    batchIdx = idx[b:min(b+batchSize,len(idx))]
    batchTarget = torch.tensor(target[batchIdx], dtype=torch.long, device = device)
    batchContext = torch.tensor(context[batchIdx], dtype=torch.long, device = device)
    
    z = model(batchContext)
    H = torch.nn.functional.cross_entropy(z,batchTarget)
    
    optimizer.zero_grad()
    H.backward()
    optimizer.step()

    if b % 10000 == 0:
        print(b, '/', len(idx), H.item())


def perplexity(model, testCorpus, word2ind, order, device, batchSize):
    target, context = extractData(testCorpus, order, word2ind)

    H = 0.
    for b in range(0,len(target),batchSize):
        batchTarget = torch.tensor(target[b:min(b+batchSize,len(target))], dtype=torch.long, device = device)
        batchContext = torch.tensor(context[b:min(b+batchSize,len(target))], dtype=torch.long, device = device)
        l = len(batchTarget)
        
        z = model(batchContext)
        H += l * torch.nn.functional.cross_entropy(z,batchTarget)

    return math.exp(H/len(target))

print(perplexity(model, testCorpus, word2ind, order, device, batchSize))


#################################################################
### Разбиване на корпуса на партиди с изречения с еднаква дължина
#################################################################

def splitCorpusInBatches(corpus, batchSize):
    minLen = min(len(s) for s in corpus)
    maxLen = max(len(s) for s in corpus)
    
    corpusBins = [ []  for _ in range(maxLen - minLen + 1) ]
    for s in corpus:
        l = len(s) - minLen
        corpusBins[l].append(s)
    
    batchCorpus = []
    for l in range(maxLen - minLen + 1):
        bin = corpusBins[l]
        idx = np.arange(len(bin), dtype='int32')
        np.random.shuffle(idx)
        for b in range(0, len(bin), batchSize):
            batch = []
            for si in range(b, min(b + batchSize, len(bin))):
                batch.append(bin[idx[si]])
            batchCorpus.append(batch)
    return batchCorpus

##########################################

class LSTMCellModel(torch.nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(LSTMCellModel, self).__init__()
        self.ii = torch.nn.Linear(embed_size, hidden_size)
        self.fi = torch.nn.Linear(embed_size, hidden_size)
        self.oi = torch.nn.Linear(embed_size, hidden_size)
        self.gi = torch.nn.Linear(embed_size, hidden_size)
        self.ih = torch.nn.Linear(hidden_size, hidden_size)
        self.fh = torch.nn.Linear(hidden_size, hidden_size)
        self.oh = torch.nn.Linear(hidden_size, hidden_size)
        self.gh = torch.nn.Linear(hidden_size, hidden_size)
    
    def forward(self, input, hc_0):
        (h_0, c_0) = hc_0
        i = torch.sigmoid(self.ii(input) + self.ih(h_0))
        f = torch.sigmoid(self.fi(input) + self.fh(h_0))
        o = torch.sigmoid(self.oi(input) + self.oh(h_0))
        g = torch.tanh(self.gi(input) + self.gh(h_0))
        c_1 = f * c_0 + i * g
        h_1 = o * torch.tanh(c_1)
        return (h_1, c_1)

class LSTMModel(torch.nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCellModel(embed_size, hidden_size)
        #self.cell = torch.nn.LSTMCell(embed_size, hidden_size)
    
    def forward(self, input):
        seq_len = input.shape[0]
        batch_size = input.shape[1]
        device = next(self.parameters()).device
        h = torch.zeros(batch_size,self.hidden_size, device = device)
        c = torch.zeros(batch_size,self.hidden_size, device = device)
        output = []
        for i in range(seq_len):
            h, c = self.cell(input[i], (h, c))
            output.append(h)
        return torch.stack(output), (h,c)

class LSTMLanguageModel(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, word2ind, unkToken):
        super(LSTMLanguageModel, self).__init__()
        self.word2ind = word2ind
        self.unkTokenIdx = word2ind[unkToken]
        #self.lstm = LSTMModel(embed_size, hidden_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size)
        self.embed = torch.nn.Embedding(len(word2ind), embed_size)
        self.projection = torch.nn.Linear(hidden_size,len(word2ind))
    
    def forward(self, source):
        ### source e списък от изречения. Всяко изречение е списък от думи
        device = next(self.parameters()).device
        batch_size = len(source)
        sents = [[self.word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        X = torch.t(torch.tensor(sents, dtype=torch.long, device=device))
        E = self.embed(X[:-1])
        output, _ = self.lstm(E)
        Z = self.projection(output.flatten(0,1))
        Y_bar = X[1:].flatten(0,1)
        H = torch.nn.functional.cross_entropy(Z,Y_bar)
        return H

#############################################################
#######   Зареждане на корпуса
#############################################################

batchSize = 32
batchCorpus = splitCorpusInBatches(trainCorpus, batchSize)


lm = LSTMLanguageModel(emb_size, hid_size, word2ind, unkToken).to(device)

optimizer = torch.optim.Adam(lm.parameters(), lr=0.01)

idx = np.arange(len(batchCorpus), dtype='int32')
np.random.shuffle(idx)

for b in range(len(idx)):
    H = lm(batchCorpus[idx[b]])
    optimizer.zero_grad()
    H.backward()
    optimizer.step()
    if b % 10 == 0:
        print(b, '/', len(idx), H.item())

def perplexity(lm, testCorpus, batchSize):
    batchCorpus = splitCorpusInBatches(testCorpus, batchSize)
    H = 0.
    c = 0
    for b in range(len(batchCorpus)):
        l = len(batchCorpus[b])*(len(batchCorpus[b][0])-1)
        c += l
        with torch.no_grad():
            H += l * lm(batchCorpus[b])
    return math.exp(H/c)

print(perplexity(lm, testCorpus, batchSize))

torch.save(lm.state_dict(), 'lstm')

for p in lm.state_dict():
    print(p,lm.state_dict()[p].size())

lm1 = LSTMLanguageModel(emb_size, hid_size, word2ind, unkToken)
lm1.load_state_dict(torch.load('lstm'))
print(perplexity(lm1, testCorpus, batchSize))

