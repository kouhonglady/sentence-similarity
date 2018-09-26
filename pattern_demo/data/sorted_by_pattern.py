#coding:utf-8

import sys
import re
import jieba as jb
from sklearn.externals import joblib
import pandas as pd
import xlrd

root_path = r'E:\study\hrg_project\environment\dataset\precision_data'
patten_path = root_path + r'/dataset/word_to_pattern.txt'
pairs_features = root_path + r"/dataset/pairs_features.txt"
final_pattern = root_path + r'/dataset/final_pattern.txt'
model_path = root_path + r'/model.txt'

def step1_cutword_to_pattern(data):
    dataset = data
    # print("step1_cutword_to_pattern :start")
    #非中文和数字字符 \u0030-\u0039 数字0-9  \u4e00-\u9fa5 所有中文字符
    pattern = re.compile(r'[^\u4e00-\u9fa5]')

    # print("start cutting setence one")
    result = []
    for q in dataset:
        q = re.sub(pattern,"",q)
        temp = jb.cut(q, cut_all=False)
        result.append(" ".join(temp))
    # print("cutting setences one has finished")
    return result


def step2_cut_top_word(data1,data2):
    sentence1 = data1
    sentence2 = data2

    result = []
    length = len(data1)
    for i in range(length):
        temp = []
        for t in sentence1[i].split():
            for k in sentence2[i].split():
                temp .append(str(t+k))
        result.append(temp)
    return result


def step3_text_filtering(feature,patterns):

    line_length = 0
    with open(pairs_features,"r") as f:
        line_length = int(f.readline())
    print(line_length)
    features_data = []
    row_ind = []
    col_ind = []
    length = len(feature)
    for i in range(length):
        for k in feature[i]:
            if k in patterns.keys():
                row_ind.append(i)
                col_ind.append(patterns[k])
                features_data.append(1)
    data = csr_matrix((features_data, (row_ind, col_ind)), shape=(length,line_length ))
    return data


from scipy.sparse import csr_matrix

def main():
    dataset = pd.read_excel('E:/study/hrg_project/environment/dataset/test.xls',header = None)
    cls = joblib.load(model_path)
    patterns = joblib.load(final_pattern)

    length = len(dataset)
    q1_result = []
    q1_original = []
    q2_result = []
    q2_original = []


    for i in range(length):
        sten1 = dataset.ix[i][2].strip('_').split("_")
        q1_original.append(sten1[0])
        # print(sten1)
        sten2 = []
        for j in range(len(sten1)):
            sten2.append(dataset.ix[i][0])
        # print(sten2)
        sentence1_words = step1_cutword_to_pattern(sten1)
        # print(sentence1_words)
        sentence2_words = step1_cutword_to_pattern(sten2)
        sen_pattern = step2_cut_top_word(sentence1_words,sentence2_words)
        # print(sen_pattern)
        data = step3_text_filtering(sen_pattern,patterns)
        #print(data)
        LogisticRegression_y_pred = cls.predict(data)
        print(LogisticRegression_y_pred)
        leng = len(LogisticRegression_y_pred)
        for k in range(leng):
            if 1 == LogisticRegression_y_pred[k]:
                q1_result.append(sten1[k])
                break;
        if k == leng-1 and len(q1_result) < i:
            q1_result.append(sten1[0])
        print("the %d  %d q2 sentence " % (i, len(q1_result)))

    for i in range(length):
        sten1 = dataset.ix[i][3].split("_")
        q2_original.append(sten1[0])
        sten2 = []
        for j in range(len(sten1)):
            sten2.append(dataset.ix[i][1])
        sentence1_words = step1_cutword_to_pattern(sten1)
        sentence2_words = step1_cutword_to_pattern(sten2)
        sen_pattern = step2_cut_top_word(sentence1_words,sentence2_words)
        data = step3_text_filtering(sen_pattern,patterns)
        LogisticRegression_y_pred = cls.predict(data)
        leng = len(LogisticRegression_y_pred)
        for k in range(leng):
            if 1 == LogisticRegression_y_pred[k]:
                q2_result.append(sten1[k])
                break;
        if k == leng-1 and len(q2_result) < i:
                q2_result.append(sten1[0])
        print("the %d  %d q1 sentence " % (i, len(q2_result)))

    ser1 = pd.Series(q1_result)
    ser2 = pd.Series(q2_result)
    ser1.append(ser2).to_csv("temp.csv")

    count = 0
    count_original = 0
    print(len(q1_result))
    print(len(q2_result))
    for i in range(len(q1_result)):
        print("%d : %s --- %s "%(count,q1_result[i],q2_result[i]))
        if q1_result[i] == q2_result[i]:
            count += 1
    for i in range(len(q1_original)):
        print("%d : %s --- %s "%(count_original,q1_original[i],q2_original[i]))
        if q1_original[i] == q2_original[i]:
            count_original += 1

    print("the result is : %.10f"%(count/(len(q1_result)+ 0.1)))
    print("the result_original  is : %.10f" % (count_original / (len(q1_result) + 0.1)))

if __name__ == '__main__':
     main()
