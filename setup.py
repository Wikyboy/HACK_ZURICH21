import json
import numpy as np

class SetupGloveDict:

    def __init__(self, categoriesDictPath):
        self.embeddings_index = {}
        path_to_glove_file = "./glove.6B/glove.6B.100d.txt"
        with open(path_to_glove_file, encoding="utf8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                self.embeddings_index[word] = coefs
        with open(categoriesDictPath, 'r') as f:
            self.categories= json.load(f)

    def genClassWordEmbeddings(self):
        gloveVectorsDict = {}
        for classNumber, wordsList in self.categories.items():
            matches = []
            for word in wordsList:
                if " " in word:
                    words = word.split(" ")
                else:
                    words = [word]
                for word in words:
                    if word.lower() in self.embeddings_index:
                        matches.append(self.embeddings_index.get(word.lower()))
            if len(matches) == 0:
                #print(f"{wordsList} had no corresponding embedding")
                pass
            else:
                gloveVectorsDict[classNumber] = np.mean(np.array(matches), axis=0)
        return gloveVectorsDict

# setup = SetupGloveDict("./categories_dict.json")
# gloveVectorsDict = setup.genClassWordEmbeddings()
# #print(gloveVectorsDict)