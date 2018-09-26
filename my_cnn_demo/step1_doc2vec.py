# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 10:59:22 2018

@author: Admin
"""

from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy 
from random import shuffle


word_path = r'./data/word.txt'
output_model_doc = r'./data/output_model_doc.model'
output_vec_doc= r'./data/output_vec_doc.vector'


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
 
        flipped = {}
 
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
 
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
 
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
 
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

sources = {word_path:'SENTENCE'}
 
sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count=1, window=15, size=100, sample=1e-4, negative=5, workers=8)
model.build_vocab(sentences.to_array())
for epoch in range(10):
    model.train(sentences.sentences_perm())

train_arrays = numpy.zeros((18293, 100))
train_labels = numpy.zeros(18293)
test_arrays = []
true_labels=[]
train_data=[]
train_lb=[]
for i in range(18293):
    if(i<=12988):
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays[i] = model.docvecs[prefix_train_neg]
        train_labels[i] = 0
    if(i>12988 and i<=18292):
        j=i-12989
        prefix_train_pos = 'TRAIN_POS_' + str(j)
        train_arrays[i]=model.docvecs[prefix_train_pos]
        train_labels[i]=1
