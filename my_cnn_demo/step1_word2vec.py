# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 09:34:27 2018

@author: Admin
"""

import pandas as pd
import re
import jieba as jb
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

    
train_data_path = r'./data/train_data.csv'
test_data_path = r'./data/test_data.csv'
word_path = r'./data/word.txt'
output_model = r'./data/output_model.model'
output_vec= r'./data/output_vec.vector'

def sentence_to_word(train_data_path,test_data_path,word_path):
     train_data = pd.read_csv(train_data_path)
     test_data = pd.read_csv(test_data_path)
     data = pd.merge(train_data,test_data,how='outer') 
     length,width = data.shape
     print(length,width)
     
     #非中文和数字字符 \u0030-\u0039 数字0-9  \u4e00-\u9fa5 所有中文字符
     pattern = re.compile(r'[^\u4e00-\u9fa5]')
     
     #将所有的句子，去除非中文字符，并且进行结巴分词，之后一行一个句子的方式保存 
     
     print("start cutting words")                                  
     with open(word_path, "w",encoding='utf-8') as f:
          i = 0;
          while i < length:
               q1= re.sub(pattern, " ",data['q1'][i])
               temp1 = jb.cut(q1, cut_all=False)
               q2= re.sub(pattern, " ",data['q2'][i])
               temp2 = jb.cut(q2, cut_all=False)
               f.write(" ".join(temp1) +"\n" )
               f.write(" ".join(temp2)+"\n")
               i += 1
          f.close()
     print("cutting setences has finished")


def wordcount(word_path):
    # 文章字符串前期处理
    with open(word_path, "r",encoding='utf-8') as f:
         string = ''
         for line in f:
              string += line.replace('\n', ' ')
         
    strl_ist = string.split(' ')
    count_dict = {}
    # 如果字典里有该单词则加1，否则添加入字典
    for str in strl_ist:
        if str in count_dict.keys():
            count_dict[str] = count_dict[str] + 1
        else:
            count_dict[str] = 1
    #按照词频从高到低排列
    count_list=sorted(count_dict.items(),key=lambda x:x[1],reverse=True)
    return count_list



def word2vec(word_path,output_model,output_vec):
     # Word2Vec函数的参数：
     # size 表示特征向量维度，默认100
     # window 表示当前词与预测词在一个句子中的最大距离
     # min_count 词频少于min_count次数的单词会被丢弃掉, 默认值为5
     print("start  word2vec")     
     model = Word2Vec(LineSentence(word_path), size=225, window=5, min_count=3,\
                     workers=multiprocessing.cpu_count())
     print("word2vec has finished")
     
     # 默认格式model
     model.save(output_model)
     # 原始c版本model
     model.wv.save_word2vec_format(output_vec, binary=False)
     
if __name__ == '__main__':
     sentence_to_word(train_data_path,test_data_path,word_path)
     #print(wordcount(word_path))
     word2vec(word_path,output_model,output_vec)