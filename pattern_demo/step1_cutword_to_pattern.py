# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 20:11:25 2018

@author: Lingxue

"""


import pandas as pd
import re
import jieba as jb


"""
   rm the characters like ,/; except chinese and numbers . cut the sentences to words,and pair the to pattern
                each lines of file is contains two sentences,and a tag 0(means two sentences have different  meaning) or 1( same meaning). 
    setence_word_q1/setence_word_q2: use the Chinese Word Segmentation API jieba to cut the q1/q2 sentences to words,and save it in file  {setence_word_q1/setence_word_q2} 
    word_to_pattern : pair one word of q1 and one word of q2 to pattern,and save all the pattern in file{word_to_pattern}
"""
    
data_path = r'./data/train_15.csv'
sentence_word_q1 = r'./data/train_15_word_q1.txt'
sentence_word_q2 = r'./data/train_15_word_q2.txt'


data = pd.read_csv(data_path)
length,width = data.shape

#非中文和数字字符 \u0030-\u0039 数字0-9  \u4e00-\u9fa5 所有中文字符
pattern = re.compile(r'[^\u0030-\u0039\u4e00-\u9fa5]')

#将所有的q1，去除非中文字符，并且进行结巴分词，之后一行一个句子的方式保存 

print("start cutting setence one")                                  
with open(sentence_word_q1, "w",encoding='utf-8') as f:
     i = 0;
     while i < length:
          q1= re.sub(pattern, "",data['q1'][i])
          temp = jb.cut(q1, cut_all=False)
          f.write(" ".join(temp) + "\n")
          i += 1
     f.close()
print("cutting setences one has finished")

#将所有的q2，去除非中文字符，并且进行结巴分词，之后一行一个句子的方式保存
print("start cutting setence two")   
with open(sentence_word_q2, "w",encoding='utf-8') as f:
     i = 0;
     while i < length:
          q2= re.sub(pattern, "",data['q2'][i])
          temp = jb.cut(q2, cut_all=False)
          f.write(" ".join(temp)+ "\n")
          i += 1
     f.close()
print("cutting setences two has finished")


                