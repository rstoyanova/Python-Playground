
import nltk
from nltk.corpus import PlaintextCorpusReader
import sys

corpus_root = 'R:/FMI/ML/Python_progr/JOURNALISM.BG/C-MassMedia'
myCorpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
fileNames = myCorpus.fileids()


def printDocuments(myCorpus, fileNames, docIDs):
    for docID in docIDs:
        text = myCorpus.words(fileNames[docID])
        print('Document ID: '+str(docID))
        for word in text:
            print(word, end=' ')
        print('\n')


class progressBar:
    def __init__(self, barWidth=80):
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
dictionary = {}
for docID, fileName in enumerate(fileNames):
    pb.tick(docID)
    text = myCorpus.words(fileName)
    for token in text:
        if not token.isalpha():
            continue
        term = token.lower()
        if term in dictionary:
            postings = dictionary[term]
            if postings[-1] < docID:
                postings.append(docID)
        else:
            dictionary[term] = [docID]
pb.stop()


dictionary[0]
