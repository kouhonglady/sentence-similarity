import tensorflow as tf
import pandas as pd
from tensorflow.contrib import learn
import numpy as np
import re
import jieba as jb
from sklearn.externals import joblib
import pickle


tf.flags.DEFINE_string("test_new",r'E:\study\hrg_project\environment\dataset\precision_data\test_new1107.xls',"the path of the test file")
tf.flags.DEFINE_string("meta_graph_path", r'E:\study\hrg_project\bigDataCompetition\bank\sentence-similarity\cnn-sentence-similarity\runs\1541636111\checkpoints\model-17020.meta', "the path of the model witch has been trained")
tf.flags.DEFINE_string("latest_checkpoint", r"E:\study\hrg_project\bigDataCompetition\bank\sentence-similarity\cnn-sentence-similarity\runs\1541636111\checkpoints", "the path of the checkpoints files")
tf.flags.DEFINE_string("vocab_processor", "./data/processed/vocab_processor.pickle", "vocab_processor.")
FLAGS = tf.flags.FLAGS

def load_model():
    dataset = pd.read_excel(FLAGS.test_new)
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

    q2_ques_top_words  = []
    q2_ans_top_words = []
    q2_mul_words = []
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    for i in range(len(q2_mul)):
        q2_temp = re.sub(pattern, " ", q2_ques_top[i])
        temp1 = jb.cut(q2_temp, cut_all=False)
        q2_ques_top_words.append(" ".join(temp1))

        q2_temp = re.sub(pattern, " ", q2_ans_top[i])
        temp1 = jb.cut(q2_temp, cut_all=False)
        q2_ans_top_words.append(" ".join(temp1))

        q2_temp = re.sub(pattern, " ", q2_mul[i])
        temp1 = jb.cut(q2_temp, cut_all=False)
        q2_mul_words.append(" ".join(temp1))


    data = {'q2_mul':q2_mul_words,'q2_ques_top':q2_ques_top_words,'q2_ans_top':q2_ans_top_words}
    print(np.shape(q2_mul_words))
    print(np.shape(q2_ques_top_words))
    print(np.shape(q2_ans_top_words))
    testdata = pd.DataFrame(data)

    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_processor)
    x_s1 = np.array(list(vocab_processor.fit_transform(testdata['q2_mul'])))
    x_s2 = np.array(list(vocab_processor.fit_transform(testdata['q2_ques_top'])))
    x = np.concatenate((x_s1, x_s2), axis=1)
    y = np.zeros([x.shape[0],2],)

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(FLAGS.meta_graph_path)
        new_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.latest_checkpoint))

        graph = tf.get_default_graph()
        feed_dict = {
            'input_x:0': x,
            'input_y:0': y,
            'dropout_keep_prob:0': 1.0
        }
        rescorse = graph.get_tensor_by_name("output/scores:0")
        rescorse_ans = sess.run(rescorse,feed_dict)

        s = pickle.dumps(rescorse_ans)
        f = open('./rescorse_ans.txt', 'wb')
        f.write(s)
        f.close()



        # rescorse_ans = joblib.load('./rescorse_ans.txt')
        # np.savetxt('./rescorse_ans.csv', rescorse_ans,delimiter=",")

        best_q2 = []
        best_q2_ans = []
        rescorse_softmax = tf.nn.softmax(rescorse_ans,1)
        print(rescorse_softmax)

        # print(rescorse_softmax)

        result = rescorse_softmax.eval()
        np.savetxt('./result.csv', result, delimiter=",")
        # total = 0
        # for i in range(length):
        #     sten1 = dataset.ix[i]['q2_ques_top'].strip('_').split("_")
        #     temp_max  = -1
        #     pos = total
        #     for item in sten1:
        #         if result[total][0]> temp_max:
        #             temp_max = result[total][0]
        #             pos = total
        #         total += 1
        #     print(pos,result[pos][0])
        #     best_q2.append(q2_ques_top[pos])
        #     best_q2_ans.append(q2_ans_top[pos])


        total = 0
        for i in range(length):
            sten1 = dataset.ix[i]['q2_ques_top'].strip('_').split("_")
            temp_max  = -1
            pos = total
            for item in sten1:
                if result[total][1]> temp_max:
                    temp_max = result[total][1]
                    pos = total
                total += 1
            print(pos,result[pos][1])
            best_q2.append(q2_ques_top[pos])
            best_q2_ans.append(q2_ans_top[pos])




        print("best_q2",len(best_q2))
        print("best_q2_ans", len(best_q2_ans))

        total_count_ans = 0
        total_count_ques = 0

        for i in range(length):
            sten1 = dataset.ix[i]['q1_ans'].strip('_').split("_")
            sten2 = dataset.ix[i]['q1'].strip('_').split("_")
            if sten1[0] == best_q2_ans[i]:
                total_count_ans +=1
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
        print("the total_count_ans: %f "%(total_count_ans / length))
        print("the total_count_ques: %f "%(total_count_ques / length))

def save_train_data():
    dataset = pd.read_excel(FLAGS.test_new)
    length = len(dataset)
    q2_ans_best = []
    q2 = []
    res = []

    for i in range(length):
        sten1 = dataset.ix[i]['q2'].strip('_').split("_")
        sten2 = dataset.ix[i]['q2_ans_top'].strip('_').split("_")
        sten3 = dataset.ix[i]['q1_ans'].strip('_').split("_")
        sten4 = dataset.ix[i]['q2_ques_top'].strip('_').split("_")
        pos = 0
        count = 0
        for item in sten2:
            q2.append(sten1[0])
            if item == sten3[0]:
                pos = count
                q2_ans_best.append(sten4[pos])
                res .append(1)
            else:
                pos = count
                q2_ans_best.append(sten4[pos])
                res.append(0)

            count += 1
    print(len(q2))
    print(len(q2_ans_best))
    print(len(res))
    data = {'q2': q2, 'q2_ans_best': q2_ans_best,"res":res}
    dataframe = pd.DataFrame(data)
    dataframe.to_csv("data_train_net.csv")


def find_base_line():
    # dataset = pd.read_excel(FLAGS.test_new)
    dataset = pd.read_excel(r"E:\study\hrg_project\backup\hrg_project20181024\test_new_pred.xlsx")
    length = len(dataset)
    q1_ans_top = []
    q2_ans_top = []


    for i in range(length):
        sten1 = dataset.ix[i]['q1_ans'].strip('_').split("_")
        q1_ans_top.append(sten1[0])

        sten2 = dataset.ix[i]['q2_ans_top'].strip('_').split("_")
        q2_ans_top.append(sten2[0])
    leng = len(q1_ans_top)
    print(len(q1_ans_top))
    print(len(q2_ans_top))
    count = 0
    for i in range(leng):
        if q2_ans_top[i] == q1_ans_top[i]:
            count += 1
    print("the count :%d"%(count))
    print("the baseline is :%f"%(count/leng))



if __name__ == "__main__":
    # load_model()
    # save_train_data()
    find_base_line()