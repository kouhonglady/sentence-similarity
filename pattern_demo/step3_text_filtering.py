# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 14:58:01 2018

@author: LingXue
"""
import pickle


patten_path = r'./data/word_to_pattern.txt'
sentence_word_q1 = r'./data/train_15_word_q1.txt'
sentence_word_q2 = r'./data/train_15_word_q2.txt'
final_pattern =  r'./data/final_pattern.txt'
pairs_features = r'./data/pairs_features.txt'

  
pattern = {}
with open(patten_path, "r",encoding='utf-8') as f:
     line = f.readline();
     while line:
          temp = line.split()
          print(temp)
          for t in temp:
               if t in pattern.keys():
                    pattern[t] += 1
               else:
                    pattern[t]=1
          line = f.readline()
     f.close()
def cmp(a,b):
     if a > b:
          return 1
     elif a < b:
          return -1
     else :
          return 0
     
#patterns = sorted(pattern, lambda x, y: cmp(x[1], y[1]))#将pattern按照出现的次数降序排序
patterns = pattern
#pos = 1  
#for k in patterns.keys():
#     patterns[k] = pos
#     pos += 1

for k, v in list(patterns.items()):
    if patterns[k] == 1: patterns.pop(k)
    else: 
          patterns[k] = pos
          pos += 1
print(pos)    

with open(final_pattern,'wb') as f:
     pickle.dump(pattern, f)
     f.close()


with open(sentence_word_q1,"r",encoding ='utf-8')as f1,open(sentence_word_q2,'r',encoding = 'utf-8') as f2,open(pairs_features,'w',encoding='utf-8') as f3:
     line1 = f1.readline()
     line2 = f2.readline()
     f3.write(str(pos) + "\n" )
     while line1 and line2:
          s = ""
          for t in line1:
               for k in line2:
                    if t+k in patterns.keys():
                         s +=" "+str(patterns[t+k])+":"+str(1)
                         
          f3.write(s+"\n")
          print(line1)
          print(line2)
          print(s+"\n")
          line1 = f1.readline()
          line2 = f2.readline()
          
     f1.close()
     f2.close()
     f3.close()
          
                         
                    
               