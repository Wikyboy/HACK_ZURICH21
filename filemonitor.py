import os
import sys
import yaml
import time
import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('brown')


class FileMonitor():

    def __init__(self, filename):
        self.filename = filename
        self.sentence = []
        self.cached_time = 0

    def filechange(self):
        time = os.stat(self.filename).st_mtime
        if time != self.cached_time:
            self.cached_time = time
            return True
        else:
            return False

    def read_file(self):
        with open(self.filename, 'r') as file:
            lines = file.readlines()
            lines_lower = [entry.strip('.').lower() for entry in lines]
            self.sentence = ' '.join(lines_lower)

    def extract_nouns(self):
        self.read_file()
        nouns = []
        for word, pos in nltk.pos_tag(nltk.word_tokenize(str(self.sentence))):
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                nouns.append(word)
        return nouns
