# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:36:26 2018

@author: Admin
"""


output_model = r'./data/output_model.model'
word_path = r'./data/word.txt'
train_feature_path = r'./data/train_feature_path.csv'
test_feature_path = r'./data/test_feature_path.csv'
train_data_path = r'./data/train_data.csv'

###读取已训练好的词向量
from gensim.models import word2vec
w2v=word2vec.Word2Vec.load(output_model)


import numpy as np
import pandas as pd


###转换成句向量
def sent2vec(s):
    words = s
    M = []
    for w in words:
        try:
            M.append(w2v.wv[w])
        except:
            continue
    M = np.array(M)
    #print(len(M),"--",len(M[0]))
    v = M.sum(axis=0)
    #print(len(v))
    return v / np.sqrt((v ** 2).sum())

def sents2vec(word_path,output_model):
     newdata = []
     newdata_dict = {}
     times = 0
     print("start convert list of words to vector")
     with open(word_path, "r",encoding='utf-8') as f:
          for line1 in f:
               line2 = f.readline()
               try:
                    newline = sent2vec(line1 + line2)
#                    print(line1)
#                    print(line2)
                    if len(newline)<225:
                         continue
                    newdata.append(newline)
                    times += 1
                    print(line1 , "-->" ,line2 , " number" ,times)
                    newdata_dict[tuple(newline)] = ''.join(line1 + line2)

               except:
                    continue
     print("end convert list of words to vector")
     df = pd.DataFrame(newdata,index = None,columns = None)
     train_data = pd.read_csv(train_data_path)
     train_length,train_width = train_data.shape
     print("save feature to %s ",train_feature_path)
     df[0:train_length-1].to_csv(train_feature_path,sep=',',header = False,index = False)
     df[train_length-1:].to_csv(test_feature_path,sep=',',header = False,index = False)
     return newdata
               
print("there are ",len(sents2vec(word_path,output_model))," pairs had got features")