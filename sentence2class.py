
import numpy as np
from numpy.linalg import norm

from sklearn.metrics import pairwise_distances as pdist

from setup import SetupGloveDict


class SentenceParser():

    def __init__(self, gloveVectorsDictPath, classes):

        setup = SetupGloveDict(gloveVectorsDictPath)
        self.fullGloveDict = setup.embeddings_index
        gloveVectorsDict = setup.genClassWordEmbeddings()

        self.gloveVectorsDict = gloveVectorsDict  # dict class # : vector
        self.gloveVectors = np.squeeze(np.array(
            [[gloveVector for gloveVector in gloveVectorsDict.values()]]))
        self.classesDict = classes
        self.numClasses = len(classes)


    def getClassesFromSentence(self, nounList):
        directMatches = self.findAllDirectMatches(nounList)
        #print(directMatches)

        return self.getClassNumsFromMatches(directMatches, nounList)

    def getClassNumsFromMatches(self, directMatches, sentenceList):
        numDirectMatches = len(directMatches)
        if numDirectMatches > 2:
            classNums = [self.classesDict[match] for match in directMatches[:2]]
        if numDirectMatches == 2:
            classNums = [self.classesDict[match] for match in directMatches]
            pass
        elif numDirectMatches == 1:
            classNum = self.classesDict[directMatches[0]]
            nearestMatch = self.findClosestMatch(classNum)
            classNums = [classNum, nearestMatch]
        elif numDirectMatches == 0:
            classNums = self.findTwoBestMatches(sentenceList)
        print(classNums)
        return list(map(str,classNums))

    def formatSentenceString(self, sentenceString):
        return sentenceString.lower().replace('-', ' ').split(" ")

    def findAllDirectMatches(self, sentenceList):
        matches = []
        for word in sentenceList:
            if word in self.classesDict:
                matches.append(word)
            else:
                for key in self.classesDict.keys():
                    if word in key.split(" "):
                        matches.append(key)
                        break

        return matches

    def findClosestMatch(self, targetClassNum):

        pairse_distance = pdist(
            self.gloveVectorsDict[targetClassNum].reshape(1,-1), self.gloveVectors, "cosine")
        sort_indeces = np.argsort(pairse_distance, axis=None)[:]
        return sort_indeces[1]

    def findTwoBestMatches(self, sentenceList):

        vectorsInSentenceList = np.array(
            [self.fullGloveDict[word] for word in sentenceList if word in self.fullGloveDict])
        if len(vectorsInSentenceList.shape) == 0:
            return [72,77] # random classes returned if no hits found
        # pairse_distance = pdist(vectorsInSentenceList,
        #                         self.gloveVectors, "cosine")
        # sort_indeces = np.argsort(pairse_distance)
        if vectorsInSentenceList.shape[0] == 1:
            pairse_distance = pdist(vectorsInSentenceList.reshape(1,-1),
                                self.gloveVectors, "cosine")
            sort_indeces = np.argsort(pairse_distance)
            sort_indeces = np.squeeze(sort_indeces[:,:2])
        else:
            pairse_distance = pdist(vectorsInSentenceList,
                                self.gloveVectors, "cosine")
            sort_indeces = np.argsort(pairse_distance)
            sort_indeces = np.squeeze(sort_indeces[:,:1])
        return sort_indeces