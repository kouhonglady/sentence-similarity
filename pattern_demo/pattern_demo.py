import re
import jieba as jb
import os
import shutil
import pickle
import pandas as pd
import numpy as np


# root_path 路径
root_path = r'E:\study\hrg_project\bigDataCompetition\bank\sentence-similarity\pattern_demo\data\data20181115_old'

# original 数据
# data_path:训练数据的路径、训练数据的格式为(q1,q2,res).
#           q1:句子一、q2:句子二、res:若句子一与句子二表示相同语义为1，不同语义为0
# word_dict :停用词文件路径、停用词格式为一行一个停用词，包括一些符号以及一些搭配
data_path = root_path + r'/original/test_new_train.xlsx'
word_dict = root_path + r'/original/Chinese_stopwords.txt'

# dataset 数据
# sentence_word_q1 :句子一(q1)分词之后保存路径，每一行保存一个句子，同一行之间的词通过空格分隔
# sentence_word_q2 :句子二(q2)分词之后保存路径，每一行保存一个句子，同一行之间的词通过空格分隔
# word_to_pattern :模式保存路径，对应的句子对，q1,q2 分词之后的结果一一对应形成模式。模式的格式为：“词1###词2”
# final_pattern :模式字典保存位置，为二进制数据。字典的格式为（string，int）即(模式，位置)
#                模式为字符串，位置为在所有的模式特征中，某个具体的模式排列在模式的第几个位置上
# pairs_features :训练数据集的句子对，被转换为稀疏矩阵储存位置。文件第一行为特征总数。从第二行开始每一行表示一个句子对的特征表示。
# result_path :运行结果保存路径
# model_path :训练完成模型的保存路径
sentence_word_q1 = root_path + r'/dataset/train_15_word_q1.txt'
sentence_word_q2 = root_path + r'/dataset/train_15_word_q2.txt'
word_to_pattern = root_path + r'/dataset/word_to_pattern.txt'
final_pattern = root_path + r'/dataset/final_pattern.txt'
pairs_features = root_path + r'/dataset/pairs_features.txt'
result_path = root_path + r'/result.txt'
model_path = root_path + r'/model_weizhong.txt'


# temp_data 数据
# 我们将输入的训练数据（q1,q1）转换为用模式表示特征的稀疏矩阵
# 这部分文件主要是用于将 pairs_features 这个文件储存的稀疏矩阵转换为 csr_matrix 需要的临时文件
# row_ind_path :稀疏矩阵中有值的位置的行坐标
# col_ind_path :稀疏矩阵中有值得位置的列坐标
# features_data_path : 稀疏矩阵有值的位置的具体值，与row_ind_path和col_ind_path相对应。
#                      但在此处句子对特征为one hot编码，由此具体值全都为1
# data_y_path :训练数据集的标签（res）保存路径，因为我们在对句子对提取模式特征的过程中可能导致，句子对没有我们想要提取的
#              特征，由此将导致训练数据集的标签位错位，我们将没有提取到特征的句子对的标签去除。得到了最后的训练标签。
features_data_path = root_path + r'\dataset\temp_data\features_data.csv'
row_ind_path = root_path + r'\dataset\temp_data\row_ind.csv'
col_ind_path = root_path + r'\dataset\temp_data\col_ind.csv'
data_y_path = root_path + r'\dataset\temp_data\data_y.csv'



# 在词对组合构成模式时，如“微信+息”和“微+信息”如果直接两个词连接构成pattern，会导致这两个模式相同即“微信息”
# 所以在两个词中间加入间隔符 the_interval_in_pattern ，上述两个模式分别是 “微信###息”“微###信息”表示不同的模式
the_interval_in_pattern = "###"

dataset = pd.read_excel(data_path)
data = dataset
data_y_all = data['res']

# M = 100000
# data = dataset[0:M]
# data_y_all = data_y_all[0:M]

#用来保存从所有数据中剔除的标签位置
line_count_list = []


Length, Width = data.shape


def step1_cut_sentence_to_words():
    print("step1_cutword_to_pattern :start")
    #非中文和数字字符 \u0030-\u0039 数字0-9  \u4e00-\u9fa5 所有中文字符
    pattern = re.compile(r'[^\u4e00-\u9fa5]')

    #将所有的q1，去除非中文字符，并且进行结巴分词，之后一行一个句子的方式保存
    print("start cutting setence one")
    with open(sentence_word_q1, "w",encoding='utf-8') as f:
        i = 0;
        while i < Length:
            q1= re.sub(pattern, "",data['q1'][i])
            # print(q1)
            temp = jb.cut(q1, cut_all=False)
            #每个词之间用空格分隔，每行保存一个句子分词之后的结果
            f.write(" ".join(temp) + "\n")
            i += 1
        f.close()

    #将所有的q2，去除非中文字符，并且进行结巴分词，之后一行一个句子的方式保存
    print("start cutting setence two")
    with open(sentence_word_q2, "w",encoding='utf-8') as f:
        i = 0;
        while i < Length:
            q2= re.sub(pattern, "",data['q2'][i])
            # print(q2)
            # 每个词之间用空格分隔，每行保存一个句子分词之后的结果
            temp = jb.cut(q2, cut_all=False)
            f.write(" ".join(temp)+ "\n")
            i += 1
        f.close()
    print("step1 end")

def rm_stopwords(file_path, word_dict):
    """
        rm stop word for {file_path}, stop words save in {word_dict} file.
        file_path: 需要移除停用词的文件
        word_dict:停用词文件，每行一个停用词
        output: file_path 文件去除停用词之后保存文件路径（此处为直接覆盖原文件）
    """

    # 从文件中将停用词读取出来，并保存到字典stop_dict中
    stop_dict = {}
    with open(word_dict,'r', encoding='UTF-8') as d:
        for word in d:
            stop_dict[word.strip("\n")] = 1
    # 如果存在临时文件将其删除
    if os.path.exists(file_path + ".tmp"):
        os.remove(file_path + ".tmp")

    print ("now remove stop words in %s." % file_path)
    # 安行删除停用词
    with open(file_path,'r', encoding='UTF-8') as f1, open(file_path + ".tmp", "w", encoding='UTF-8') as f2:
        for line in f1:
            tmp_list = []  # 将非停用词保存下来
            words = line.split()
            #	        print(line)
            for word in words:
                if word not in stop_dict:
                    tmp_list.append(word)
            words_without_stop = " ".join(tmp_list)
            to_write = words_without_stop + "\n"
            f2.write(to_write)
        f1.close()
        f2.close()
    # 将去除停用词的pattern文件重新到原路径
    shutil.move(file_path + ".tmp", file_path)
    print ("stop words in %s has been removed." % file_path)



def step2_word_top_pattern():
    print ("step2_word_top_pattern: start.")
    #q1,q2 分词之后，分别两两组合，形成pattern
    line_count = 0
    with open(sentence_word_q1,"r",encoding ='utf-8')as f1,open(sentence_word_q2,'r',encoding = 'utf-8') as f2,open(word_to_pattern,'w',encoding='utf-8') as f3:
        line1 = f1.readline()
        line2 = f2.readline()
        while line1 and line2:
            temp=""
            for t in line1.split():
                for k in line2.split():
                    temp = temp+" "+t+ the_interval_in_pattern +k
            if temp is not "":
                f3.write(temp + '\n')
            else :
                #在模式组合过程中，会出现空串现象，因此无法提取到任何特征
                # 没有提取到特征的训练样本，将被直接抛弃，由此将其标签记录
                line_count_list.append(line_count)
            line_count += 1
            line1 = f1.readline()
            line2 = f2.readline()
        f1.close()
        f2.close()
        f3.close()
    # rm_stopwords(word_to_pattern, word_dict)
    print("step2 end.")



def step3_pattern_filtering():
    print("step3_text_filtering: start to change sentences pairs to features")
    pattern = {}

    with open(word_to_pattern, "r",encoding='utf-8') as f:
        line = f.readline()
        while line:
            temp = line.split()
            for t in temp:
                if t in pattern.keys():
                    pattern[t] += 1
                else:
                    pattern[t]=1
            line = f.readline()
        f.close()

    patterns = pattern
    # pos = 1
    # for k in patterns.keys():
    #     patterns[k] = pos
    #     pos += 1
    pos = 0
    print(len(patterns))
    for k, v in list(patterns.items()):
        if patterns[k] <= 1:
            # print(k,v)
            patterns.pop(k)
        else:
            patterns[k] = pos
            pos += 1

    with open(final_pattern,'wb') as f:
        pickle.dump(pattern, f)
        f.close()

    with open(word_to_pattern,"r",encoding ='utf-8')as f1,open(pairs_features,'w',encoding='utf-8') as f3:
        line1 = f1.readline().strip().split()
        f3.write(str(pos) + "\n" )
        line_count = 0
        while line1:
            # print(line1)
            s = ""
            for t in line1:
                if t in patterns.keys():
                    s +=" "+str(patterns[t])+":"+str(1)
            if s  is not "":
                f3.write(s+"\n" )
            else:
                #在模式组合过程中，会出现空串现象，因此无法提取到任何特征
                # 没有提取到特征的训练样本，将被直接抛弃，由此将其标签记录
                line_count_list.append(line_count)
            line_count += 1
            line1 = f1.readline().strip().split()
        f1.close()
        f3.close()
        # print(line_count_list)
        data_y = data_y_all.drop(line_count_list)
        np.savetxt(data_y_path, data_y,delimiter=",")
    print("step3 end")



def step4_patter_to_feature():
    print("step4_patter_to_feature : start")
    features_data = []
    row_ind = []
    col_ind = []

    with open(pairs_features, 'r') as f:
        sizes_of_features = int(f.readline())
        print(sizes_of_features)
        line = f.readline().strip()
        line_count = 0
        while line:
            if line_count % 200 == 0:
                print("step :%d "%line_count)
            # print(line_count)
            for k in line.split():
                temp = int(k.split(':')[0])

                if temp not in col_ind:
                    row_ind.append(line_count)
                    col_ind.append(temp)
                    features_data.append(1)
                else:
                    pos = col_ind.index(temp)
                    if row_ind[pos] != line_count:
                        row_ind.append(line_count)
                        col_ind.append(temp)
                        features_data.append(1)
            line_count += 1
            line = f.readline().strip()
    f.close()

    np.savetxt(features_data_path,features_data,delimiter=",")
    np.savetxt(row_ind_path, row_ind,delimiter=",")
    np.savetxt(col_ind_path, col_ind,delimiter=",")
    print("step4 end")

from scipy.sparse import csr_matrix

# csr_matrix
# (data, (row_ind, col_ind)

def step5_lr():
    # import pprint, pickle
    #
    # pkl_file = open(final_pattern, 'rb')
    #
    # data1 = pickle.load(pkl_file)
    # pprint.pprint(data1)
    # pkl_file.close()

    print("step5_lr: start")
    with open(pairs_features, 'r') as f:
        sizes_of_features = int(f.readline())

    data_y = np.loadtxt(data_y_path, delimiter=",")
    features_data = np.loadtxt(features_data_path,delimiter=",")
    row_ind = np.loadtxt(row_ind_path,delimiter=",")
    col_ind = np.loadtxt(col_ind_path,delimiter=",")

    data = csr_matrix((features_data, (row_ind, col_ind)), shape=(data_y.shape[0], sizes_of_features))

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn import preprocessing

    temp_data = preprocessing.maxabs_scale(data)

    # print(data.shape)

    for step in range(1):
        # cls = LogisticRegression(penalty= 'l2')
        cls = LogisticRegression(penalty='l2', class_weight={0: 0.50, 1: 0.50}, solver='liblinear', random_state=1763)

        x_train, x_test, y_train, y_test = train_test_split(temp_data, data_y, test_size = 0.3, random_state=1729)
        # 选择模型
        # 把数据交给模型训练

        cls.fit(x_train, y_train)
        LogisticRegression_y_pred = cls.predict(x_test)



        print(y_test)
        print(LogisticRegression_y_pred)
        np.savetxt(root_path + r"/LogisticRegression_y_pred.csv", LogisticRegression_y_pred, delimiter=",")
        print("LogisticRegression_y_pred step :%d  F1_score: %.10f" %(step ,f1_score(y_test, LogisticRegression_y_pred)))

        # f = open(result_path,'w+',encoding = 'utf-8')
        #
        # f.write( "step :%d  F1_score: %.10f" %(step,f1_score(y_test, LogisticRegression_y_pred)))
        #
        # clf = SVC(gamma='auto')
        # clf.fit(x_train, y_train)
        # svm_y_pred = clf.predict(x_test)
        # print("svm_y_pred step :%d  F1_score: %.10f" % (step, f1_score(y_test, svm_y_pred)))


    s = pickle.dumps(cls)
    f = open(model_path, 'wb')
    f.write(s)
    f.close()



if __name__ == '__main__':

    step1_cut_sentence_to_words()
    step2_word_top_pattern()
    step3_pattern_filtering()
    step4_patter_to_feature()
    step5_lr()

