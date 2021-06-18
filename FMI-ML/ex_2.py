#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################

### Упражнение 2
###
### За да работи програмата трябва да се свали корпус от публицистични текстове за Югоизточна Европа,
### предоставен за некомерсиално ползване от Института за български език - БАН
### Корпусът може да бъде свален от:
### http://dcl.bas.bg/BulNC-registration/dl.php?dl=feeds/JOURNALISM.BG.zip
### Архивът трябва да се разархивира в директорията, в която е програмата.

### Преди да се стартира програмата на питон е необходимо да се активира съответното обкръжение с командата:
### conda activate tii

import nltk
from nltk.corpus import PlaintextCorpusReader
import sys

corpus_root = 'JOURNALISM.BG/C-MassMedia'
myCorpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
fileNames = myCorpus.fileids()

def printDocuments(myCorpus,fileNames,docIDs):
    for docID in docIDs:
        text = myCorpus.words(fileNames[docID])
        print('Document ID: '+str(docID))
        for word in text:
            print(word,end=' ')
        print('\n')


class progressBar:
    def __init__(self ,barWidth = 80):
        self.barWidth = barWidth
        self.period = None

    def start(self, count):
        self.period = int(count / self.barWidth)
        sys.stdout.write("["+(" " * self.barWidth)+"]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth+1))

    def tick(self, item):
        if item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()

    def stop(self):
        sys.stdout.write("]\n")


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
            postings = dictionary[term]
            if postings[-1] < docID:
                postings.append(docID)
        else:
            dictionary[term] = [docID]
pb.stop()

def intersect(p1,p2):
    i1=0
    i2=0
    answer=[]
    while i1<len(p1) and i2<len(p2):
        if p1[i1] == p2[i2]:
            answer.append(p1[i1])
            i1 += 1
            i2 += 1
        elif p1[i1] < p2[i2]:
            i1 += 1
        else:
            i2 += 1
    return answer

def intersectLists(andLists):
    if not andLists: return []
    postingLists = sorted(andLists, key = lambda x : len(x))
    answer = postingLists[0]
    for i in range(1,len(postingLists)):
        answer = intersect(answer,postingLists[i])
    return answer

def andSearch(dictionary, termList):
    dicItems = [ dictionary[term] for term in termList if term in dictionary ]
    return intersectLists(dicItems)

printDocuments(myCorpus, fileNames, andSearch(dictionary, ['културен','обмен','театър']))

def union(p1,p2):
    i1 = 0
    i2 = 0
    answer = []
    while i1 < len(p1) or i2 < len(p2):
        if i2 == len(p2) or (i1 < len(p1) and p1[i1] < p2[i2]):
            answer.append(p1[i1])
            i1 += 1
        elif i1 == len(p1) or (i2 < len(p2) and p2[i2] < p1[i1]):
            answer.append(p2[i2])
            i2 += 1
        else:
            answer.append(p1[i1])
            i1 += 1
            i2 += 1
    return answer

class Trie:
    def __init__(self):
        self.stateTransitions = [{}]
        self.finalStates = set([])

    def traverse(self,word,initialState = 0):
        s = initialState
        i = 0
        while i < len(word) and word[i] in self.stateTransitions[s]:
            s = self.stateTransitions[s][word[i]]
            i += 1
        return s, i

    def isFinal(self, s):
        return s in self.finalStates

    def inTrie(self, word):
        s, i = self.traverse(word)
        return i == len(word) and self.isFinal(s)

    def addWord(self,word):
        s, i = self.traverse(word)
        while i < len(word):
            newState = len(self.stateTransitions)
            self.stateTransitions.append({})
            st = self.stateTransitions[s]
            st[word[i]] = newState
            s = newState
            i += 1
        self.finalStates.add(s)

    def getWordsFromState(self,s):
        answer = []
        if s in self.finalStates:
            answer.append('')
        st = self.stateTransitions[s]
        for (char,nextstate) in st.items():
            answer += [ char + word for word in self.getWordsFromState(nextstate) ]
        return answer

    def getWordsWithPrefix(self,prefix):
        s,i = self.traverse(prefix)
        if i != len(prefix):
            return []
        else:
            return [ prefix + word for word in self.getWordsFromState(s) ]

dictionaryTrie=Trie()
for term in list(dictionary):
    dictionaryTrie.addWord(term)

def andSearchWithWildCards(dictionary,dictionaryTrie,patternList):
    items = []
    for pattern in patternList:
        orList = []
        for term in dictionaryTrie.getWordsWithPrefix(pattern):
            orList = union(orList,dictionary[term])
        items.append(orList)
    return intersectLists(items)

andSearchWithWildCards(dictionary, dictionaryTrie, ['култур','обмен','теат'])
