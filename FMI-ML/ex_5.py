#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################

### Упражнение 5
###
### За да работи програмата трябва корпуса от публицистични текстове за Югоизточна Европа,
### да се намира разархивиран в директорията, в която е програмата (виж упражнение 2).

### Преди да се стартира програмата е необходимо да се активира съответното обкръжение с командата:
### conda activate tii

import nltk
from nltk.corpus import PlaintextCorpusReader
import sys
import random
import math
import matplotlib.pyplot as plt

corpus_root = 'JOURNALISM.BG/C-MassMedia'
myCorpus = PlaintextCorpusReader(corpus_root, '.*\.txt')

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

fullSentCorpus = [ [startToken] + [w.lower() for w in sent] + [endToken] for sent in myCorpus.sents()]
testDevSentCorpus, trainSentCorpus = splitSentCorpus(fullSentCorpus)
testSentCorpus, devSentCorpus = splitSentCorpus(testDevSentCorpus, testFraction = 0.5)

class MarkovModel:
    def __init__(self, corpus, K, dictionaryLimit = 50000, startToken = '<START>', endToken = '<END>', unkToken = '<UNK>'):
        self.K = K
        self.startToken = startToken
        self.endToken = endToken
        self.unkToken = unkToken
        self.kgrams ={}
        self.extractMonograms(corpus,dictionaryLimit)
        for k in range(2,K+1):
            self.extractKgrams(corpus,k)
        self.Tc = {}
        for context in self.kgrams:
            self.Tc[context] = sum( self.kgrams[context][v] for v in self.kgrams[context])

    def extractMonograms(self, corpus,limit):
        pb = progressBar()
        pb.start(len(corpus))
        dictionary = {}
        for sent in corpus:
            pb.tick()
            for i in range(1,len(sent)):
                w = sent[i]
                if w not in dictionary:
                    dictionary[w] = 0
                dictionary[w] += 1
        L = sorted([(w,dictionary[w]) for w in dictionary], key = lambda x: x[1] , reverse=True)
        if limit > len(L): limit = len(L)
        mono = { w:c for (w,c) in L[:limit] }
        sumUnk = sum( c for (w,c) in L[limit:] )
        mono[self.unkToken] = sumUnk
        self.kgrams[tuple()] = mono
        pb.stop()

    def substituteUnkownWords(self, sentence):
        return [ w if w in self.kgrams[tuple()] else self.unkToken for w in sentence]

    def getContext(self, sent, k, i):
        if i-k+1 >= 0:
            context = sent[i-k+1:i]
        else:
            context = [self.startToken] * (k-i-1) + sent[:i]
        return tuple(context)

    def extractKgrams(self, corpus, k):
        pb = progressBar()
        pb.start(len(corpus))
        for s in corpus:
            pb.tick()
            sent = self.substituteUnkownWords(s)
            for i in range(1,len(sent)):
                w = sent[i]
                context = self.getContext(sent,k,i)
                if context not in self.kgrams: self.kgrams[context] = {}
                if w not in self.kgrams[context]: self.kgrams[context][w] = 0
                self.kgrams[context][w] += 1
        pb.stop()

    def probMLE(self, w ,context):
        if context not in self.kgrams:
            return 0.0
        elif w not in self.kgrams[context]:
            return 0.0
        else:
            return self.kgrams[context][w] / self.Tc[context]

    def prob(self, w, context, alpha):
        if context:
            return alpha * self.probMLE(w,context) + (1-alpha) * self.prob(w,context[1:],alpha)
        else:
            return self.probMLE(w,context)

    def sentenceLogProbability(self, s, alpha):
        sent = self.substituteUnkownWords(s)
        return sum(math.log(self.prob(sent[i],self.getContext(sent,self.K,i),alpha),2) for i in range(1,len(sent)))

    def bestContinuation(self, sentence, alpha, l):
        context = self.getContext(self.substituteUnkownWords(sentence), self.K, len(sentence))
        for k in range(0,self.K):
            if context[k:] in self.kgrams and len(self.kgrams[context[k:]]) >= l:
                candidates = self.kgrams[context[k:]]
                break
        L = [(w,self.prob(w,context,alpha)) for w in candidates ]
        return sorted(L, key = lambda x: x[1], reverse=True)[:l]

    def perplexity(self, corpus, alpha):
        pb = progressBar()
        pb.start(len(corpus))
        m = sum(len(s)-1 for s in corpus)
        crossEntropy=0.0
        for s in corpus:
            pb.tick()
            crossEntropy -= self.sentenceLogProbability(s,alpha)
        crossEntropyRate = crossEntropy / m
        pb.stop()
        return 2 ** crossEntropyRate

M1 = MarkovModel(trainSentCorpus,1)
print('Перплексията на монограмния модел върху dev е: '+str(M1.perplexity(devSentCorpus,0)))
print('Перплексията на монограмния модел върху test е: '+str(M1.perplexity(testSentCorpus,0)))
M1.bestContinuation(['<START>', 'от', 'днес', 'до', 'края', 'на'],0,10)

M2 = MarkovModel(trainSentCorpus,2)
M2.bestContinuation(['<START>', 'от', 'днес', 'до', 'края', 'на'],0.9, 10)

M3 = MarkovModel(trainSentCorpus,3)
M3.bestContinuation(['<START>', 'от', 'днес', 'до', 'края', 'на'],0.6, 10)

M2.bestContinuation(['<START>', 'в', 'софия', 'се'],0.9, 10)
M3.bestContinuation(['<START>', 'в', 'софия', 'се'],0.6, 10)

alpha=[0.01]+[0.1*k for k in range(1,10)]+[0.99]
perp2 = [M2.perplexity(devSentCorpus,a) for a in alpha]
perp3 = [M3.perplexity(devSentCorpus,a) for a in alpha]

plt.plot(alpha,perp2)
plt.plot(alpha,perp3)
plt.ylabel('Перплексия')
plt.xlabel('Алфа')
plt.legend(['Биграмен','Триграмен'])
plt.show()

print('Перплексията на биграмния модел при алфа=0.9 върху dev е: '+str(M2.perplexity(devSentCorpus,0.9)))
print('Перплексията на биграмния модел при алфа=0.9 върху test е: '+str(M2.perplexity(testSentCorpus,0.9)))
print('Перплексията на триграмния модел при алфа=0.6 върху dev е: '+str(M3.perplexity(devSentCorpus,0.6)))
print('Перплексията на триграмния модел при алфа=0.6 върху test е: '+str(M3.perplexity(testSentCorpus,0.6)))


corpus1984 = [ [startToken] + [w.lower() for w in sent] + [endToken] for sent in PlaintextCorpusReader('.', '1984.txt').sents() ]
M3.perplexity(corpus1984,0.6)

corpusNovinite = [ [startToken] + [w.lower() for w in sent] + [endToken] for sent in PlaintextCorpusReader('.', 'novinite.txt').sents() ]
M3.perplexity(corpusNovinite,0.6)
