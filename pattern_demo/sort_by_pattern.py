import re
import jieba as jb
import os
import shutil
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.externals import joblib


# root_path 路径
root_path = r'E:\study\hrg_project\bigDataCompetition\bank\sentence-similarity\pattern_demo\data\data20181115_old'
# test_new 路径
test_new = root_path+ r'\original\test_new_pred.xlsx'


final_pattern = root_path + r'/dataset/final_pattern.txt'
model_path = root_path + r'/model_weizhong.txt'
features_data_path = root_path + r'\dataset\temp_data\features_data.csv'
row_ind_path = root_path + r'\dataset\temp_data\row_ind.csv'
col_ind_path = root_path + r'\dataset\temp_data\col_ind.csv'
data_y_path = root_path + r'\dataset\temp_data\data_y.csv'
pairs_features = root_path + r'/dataset/pairs_features.txt'

the_interval_in_pattern = "###"



def load_model():
    dataset = pd.read_excel(test_new)
    length = len(dataset)
    q2_ques_top = []
    q2_ans_top = []
    q2_mul = []

    for i in range(length):
        sten1 = dataset.ix[i]['q2_ques_top'].strip('_').split("_")
        for item in sten1:
            q2_ques_top.append(item)

        sten2 = dataset.ix[i]['q2_ans_top'].strip('_').split("_")
        # print(sten2)
        for item in sten2:
            q2_ans_top.append(item)

        sten3 = dataset.ix[i]['q2'].strip('_').split("_")
        size = len(sten2)
        for i in range(size):
            q2_mul.append(sten3[0])

    print(len(q2_ques_top))
    print(len(q2_ans_top))
    print(len(q2_mul))


    q2_and_q2_ques_top_words = []
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    for i in range(len(q2_mul)):
        q2_temp = re.sub(pattern, " ", q2_ques_top[i])
        temp1 = jb.cut(q2_temp, cut_all=False)
        q2_ques_top_words = (" ".join(temp1))

        q2_temp = re.sub(pattern, " ", q2_mul[i])
        temp2 = jb.cut(q2_temp, cut_all=False)
        q2_mul_words = (" ".join(temp2))

        temp = []
        for t in q2_mul_words.split():
            for k in q2_ques_top_words.split():
                temp.append(t + the_interval_in_pattern + k )
        q2_and_q2_ques_top_words.append(temp)

    print(len(q2_and_q2_ques_top_words))

    print("step4_patter_to_feature : start")
    features_data = []
    row_ind = []
    col_ind = []

    import pprint, pickle
    pkl_file = open(final_pattern, 'rb')
    patterns = pickle.load(pkl_file)
    # pprint.pprint(patterns)
    pkl_file.close()

    length_all_pairs = len(q2_and_q2_ques_top_words)
    for line_count in range(length_all_pairs):
        line_item = []
        for item in q2_and_q2_ques_top_words[line_count]:
            if item in patterns.keys():
                if item not in line_item:
                    line_item.append(item)
                    row_ind.append(line_count)
                    col_ind.append(patterns[item])
                    features_data.append(1)
        line_count += 1

    np.savetxt(features_data_path, features_data, delimiter=",")
    np.savetxt(row_ind_path, row_ind, delimiter=",")
    np.savetxt(col_ind_path, col_ind, delimiter=",")

    print("step5_lr: start")
    with open(pairs_features, 'r') as f:
        sizes_of_features = int(f.readline())

    features_data = np.loadtxt(features_data_path,delimiter=",")
    row_ind = np.loadtxt(row_ind_path,delimiter=",")
    col_ind = np.loadtxt(col_ind_path,delimiter=",")

    data = csr_matrix((features_data, (row_ind, col_ind)), shape=(length_all_pairs, sizes_of_features))

    classification = joblib.load(model_path)
    LogisticRegression_y_pred = classification.predict(data)
    print(LogisticRegression_y_pred)
    np.savetxt(root_path + r"/LogisticRegression_y_pred.csv", LogisticRegression_y_pred, delimiter=",")

    best_q2 = []
    best_q2_ans = []
    total = 0
    for i in range(length):
        sten1 = dataset.ix[i]['q2_ques_top'].strip('_').split("_")
        temp_max = -1
        pos = total
        for item in sten1:
            if LogisticRegression_y_pred[total] > temp_max:
                temp_max = LogisticRegression_y_pred[total]
                pos = total
            total += 1
        print(pos, LogisticRegression_y_pred[pos])
        best_q2.append(q2_ques_top[pos])
        best_q2_ans.append(q2_ans_top[pos])

    print("best_q2", len(best_q2))
    print("best_q2_ans", len(best_q2_ans))

    total_count_ans = 0
    total_count_ques = 0

    for i in range(length):
        sten1 = dataset.ix[i]['q1_ans'].strip('_').split("_")
        sten2 = dataset.ix[i]['q1'].strip('_').split("_")
        if sten1[0] == best_q2_ans[i]:
            total_count_ans += 1
            # print(sten1[0])
            # print(best_q2_ans[i])
        else:
            print(sten1[0])
            # print(best_q2_ans[i])
        if sten2[0] == best_q2[i]:
            total_count_ques += 1
    print(total_count_ans)
    print(total_count_ques)
    print(length)
    print("the total_count_ans: %f " % (total_count_ans / length))
    print("the total_count_ques: %f " % (total_count_ques / length))


if __name__ == "__main__":
    load_model()