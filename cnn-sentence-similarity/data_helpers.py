import numpy as np
import re
import pandas as pd
import jieba as jb



def sentence_to_word(train_data_path,train_data_s1_path,train_data_s2_path):
    train_data = pd.read_excel(train_data_path)
    print(train_data.shape)
    # 非中文和数字字符 \u0030-\u0039 数字0-9  \u4e00-\u9fa5 所有中文字符
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    # 将所有的句子，去除非中文字符，并且进行结巴分词，之后一行一个句子的方式保存
    print("start cutting train  words")
    with open(train_data_s1_path, "w", encoding='utf-8') as f1,open(train_data_s2_path, "w", encoding='utf-8') as f2:
        i = 0;
        while i < train_data.shape[0]:
            q1 = re.sub(pattern, " ", train_data['q1'][i])
            temp1 = jb.cut(q1, cut_all=False)
            f1.write(" ".join(temp1) + "\n")
            q2 = re.sub(pattern, " ", train_data['q2'][i])
            temp2 = jb.cut(q2, cut_all=False)
            f2.write(" ".join(temp2) + "\n")
            i += 1
    print("cutting train setences has finished")

def load_data_and_labels(train_data_s1_path, train_data_s2_path,train_data_path):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    train_data_s1 = list(open(train_data_s1_path, "r", encoding='utf-8').readlines())
    x_text_s1 = [s.strip() for s in train_data_s1]
    train_data_s2 = list(open(train_data_s2_path, "r", encoding='utf-8').readlines())
    x_text_s2 = [s.strip() for s in train_data_s2]

    train_data = pd.read_excel(train_data_path)
    y_train = train_data['res']
    y_temp  = [[0, 1] if t is 1 else [1, 0] for t in y_train]
    y = np.concatenate([y_temp, ], 0)
    return [x_text_s1,x_text_s2, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
