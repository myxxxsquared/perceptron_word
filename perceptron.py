#!/usr/bin/env python3

import numpy as np
import os
import sys
import json
from collections import defaultdict
from itertools import count
import random

class increase_defaultdict(defaultdict):
    def __init__(self):
        super().__init__(lambda: len(self))

LABEL_BEGIN = 0
LABEL_MIDDLE = 1
LABEL_END = 2
LABEL_SINGLE = 3

class PerceptronModel:
    def __init__(self, nfeatures, noutputs):
        self.nfeatures = nfeatures
        self.noutputs = noutputs
        self.dictfeatures = [increase_defaultdict() for _ in range(self.nfeatures)]
        self.weights = None
    def init_weights(self):
        self.weights = []
        for feature in self.dictfeatures:
            self.weights.append(np.zeros(
                shape=[len(feature), self.noutputs],
                dtype=np.int32))
    def load(self, path):
        dictfile = os.path.join(path, "dict")
        weightfile = os.path.join(path, "weight")
        self.weights = []
        self.dictfeatures = [increase_defaultdict() for _ in range(self.nfeatures)]
        for i in range(self.nfeatures):
            with open("{}_{:03d}".format(dictfile, i), 'r', encoding='utf8') as f:
                self.dictfeatures[i].update(json.load(f))
            with open("{}_{:03d}".format(weightfile, i), 'rb') as f:
                self.weights.append(np.load(f))
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        dictfile = os.path.join(path, "dict")
        weightfile = os.path.join(path, "weight")
        for i in range(self.nfeatures):
            with open("{}_{:03d}".format(dictfile, i), 'w', encoding='utf8') as f:
                json.dump(dict(self.dictfeatures[i]), f)
            with open("{}_{:03d}".format(weightfile, i), 'wb') as f:
                np.save(f, self.weights[i])
    def init_with_file(self, file):
        for features, _ in file:
            for feature in features:
                for f, d in zip(feature, self.dictfeatures):
                    _ = d[f]
    def map_file(self, file):
        filecontent = []
        for features, labels in file:
            newfeatures = []
            for feature in features:
                newfeature = []
                for f, d in zip(feature, self.dictfeatures):
                    newfeature.append(d[f])
                newfeatures.append(newfeature)
            filecontent.append((newfeatures, labels))
        return filecontent
    def learn_file(self, file):
        for features, labels in file:
            for feature, label in zip(features, labels):
                result = np.zeros(shape=[self.noutputs], dtype=np.int32)
                for f, d in zip(feature, self.weights):
                    result += d[f]
                for i in range(4):
                    if result[i] >= 0 and label != i:
                        for f, d in zip(feature, self.weights):
                            d[f][i] -= 1
                    if result[i] < 0 and label == i:
                        for f, d in zip(feature, self.weights):
                            d[f][i] += 1
    def predict_file(self, file):
        result = []
        for features, _ in file:
            labels = []
            for feature in features:
                r = np.zeros(shape=[self.noutputs], dtype=np.int32)
                for f, d in zip(feature, self.weights):
                    r += d[f]
                label = np.argmax(r)
                labels.append(label)
            result.append(labels)
        return result


def readfile(filename):
    filecontent = []
    with open(filename, encoding='utf8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            characters = [ '$' ]
            labels = [ True ]
            nextnew = True
            for c in line:
                if c == ' ':
                    nextnew = True
                else:
                    characters.append(c)
                    labels.append(nextnew)
                    nextnew = False
            characters.append('^')
            labels.append(True)
            features = []
            fourlabels = []
            for i in range(1, len(labels)-1):
                features.append((
                    characters[i],
                    characters[i-1],
                    characters[i+1],
                    characters[i-1]+characters[i],
                    characters[i]+characters[i+1],
                    characters[i-1]+characters[i+1]))
                if labels[i]:
                    if labels[i+1]:
                        fourlabels.append(LABEL_SINGLE)
                    else:
                        fourlabels.append(LABEL_BEGIN)
                else:
                    if labels[i+1]:
                        fourlabels.append(LABEL_END)
                    else:
                        fourlabels.append(LABEL_MIDDLE)
            filecontent.append((features, fourlabels))
    return filecontent

def main():
    file = readfile("train.txt")
    filetest = readfile("test.answer.txt")
    model = PerceptronModel(6, 4)
    model.init_with_file(file)
    model.init_with_file(filetest)
    model.init_weights()
    file = model.map_file(file)
    filetest = model.map_file(filetest)

    for i in range(100):
        print(i)

        random.shuffle(file)
        model.learn_file(file)

        model.save(os.path.join(".", "result"))
        labels = model.predict_file(filetest)
        sumnum = 0
        sumcorrect = 0
        for (_, ols), ls in zip(filetest, labels):
            for ol, l in zip(ols, ls):
                if ol == l:
                    sumcorrect += 1
                sumnum += 1
        print(sumnum, sumcorrect, sumcorrect/sumnum)


if __name__ == '__main__':
    main()
