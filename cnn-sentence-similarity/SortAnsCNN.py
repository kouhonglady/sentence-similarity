# coding=utf-8

import sys
import tensorflow as tf
import pandas as pd
from tensorflow.contrib import learn
import numpy as np
import re
import jieba as jb
from sklearn.externals import joblib
import pickle


tf.flags.DEFINE_string("meta_graph_path", r'E:\study\hrg_project\bigDataCompetition\bank\sentence-similarity\cnn-sentence-similarity\runs\1541636111\checkpoints\model-17020.meta', "the path of the model witch has been trained")
tf.flags.DEFINE_string("latest_checkpoint", r"E:\study\hrg_project\bigDataCompetition\bank\sentence-similarity\cnn-sentence-similarity\runs\1541636111\checkpoints", "the path of the checkpoints files")
tf.flags.DEFINE_string("vocab_processor", r"E:/study/hrg_project/bigDataCompetition/bank/sentence-similarity/cnn-sentence-similarity/data/processed/vocab_processor.pickle", "vocab_processor.")
FLAGS = tf.flags.FLAGS

def sort_by_cnn_func(question,all_top_data):
    # question = sys.argv[1]
    # all_top_data = sys.argv[2]
    all_top_data_list = all_top_data.strip('#').split("#")
    all_top_data_list_q = []
    all_top_data_list_q2_ques_top_words = []
    all_top_data_list_q2_ans_top_words = []
    q2_question = []
    q2_answer = []

    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    question_list = " ".join(jb.cut( re.sub(pattern, " ", question), cut_all=False))
    for item in all_top_data_list:
        pair = item.strip("_").split("_")
        if len(pair) == 2:
            q2_question.append(pair[0])
            q2_answer.append(pair[1])

            all_top_data_list_q.append(question_list)
            all_top_data_list_q2_ques_top_words.append(" ".join(jb.cut( re.sub(pattern, " ", pair[0]), cut_all=False)))
            all_top_data_list_q2_ans_top_words.append(" ".join(jb.cut( re.sub(pattern, " ", pair[1]), cut_all=False)))

    data = {'q2':all_top_data_list_q,'q2_ques_top':all_top_data_list_q2_ques_top_words,'q2_ans_top':all_top_data_list_q2_ans_top_words}
    print(np.shape(all_top_data_list_q))
    print(np.shape(all_top_data_list_q2_ques_top_words))
    print(np.shape(all_top_data_list_q2_ans_top_words))
    testdata = pd.DataFrame(data)

    print(testdata['q2'])
    print(testdata['q2_ques_top'])


    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_processor)
    x_s1 = np.array(list(vocab_processor.fit_transform(testdata['q2'])))
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
        rescorse_ans = sess.run(rescorse, feed_dict)

        rescorse_softmax = tf.nn.softmax(rescorse_ans, 1)
        # print(rescorse_softmax)
        result = rescorse_softmax.eval()
        # print(result)
        data_result = {'q2_ques_top':q2_question,'q2_ans_top':q2_answer,"score":result[:,1]}
        data_result_dataframe =  pd.DataFrame(data_result)
        sorded_data_result_dataframe = data_result_dataframe.sort_values(by='score',ascending = False)

        for index, row in sorded_data_result_dataframe.iterrows():
            print(row['q2_ques_top'],"_",row['q2_ans_top'])

def testfunc():
    print("对吧拜托")

class Singleton(object):
    _instance = None

    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kw)
        return cls._instance

class MyClass(Singleton):
    def __init__(self):
        self.question = ""
        self.all_top_data = ""
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_processor)
        self.new_saver = tf.train.import_meta_graph(FLAGS.meta_graph_path)
        self.graph = tf.get_default_graph()
        self.q2_question = None
        self.q2_answer = None
        self.data = None
        self.x = None
        self.y = None


    def prepare_data(self):
        all_top_data_list = self.all_top_data.strip('#').split("#")
        all_top_data_list_q = []
        all_top_data_list_q2_ques_top_words = []
        all_top_data_list_q2_ans_top_words = []
        q2_question = []
        q2_answer = []

        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        question_list = " ".join(jb.cut(re.sub(pattern, " ", self.question), cut_all=False))
        for item in all_top_data_list:
            pair = item.strip("_").split("_")
            if len(pair) == 2:
                q2_question.append(pair[0])
                q2_answer.append(pair[1])

                all_top_data_list_q.append(question_list)
                all_top_data_list_q2_ques_top_words.append(
                    " ".join(jb.cut(re.sub(pattern, " ", pair[0]), cut_all=False)))
                all_top_data_list_q2_ans_top_words.append(
                    " ".join(jb.cut(re.sub(pattern, " ", pair[1]), cut_all=False)))

        data = {'q2': all_top_data_list_q, 'q2_ques_top': all_top_data_list_q2_ques_top_words,
                'q2_ans_top': all_top_data_list_q2_ans_top_words}
        print(np.shape(all_top_data_list_q))
        print(np.shape(all_top_data_list_q2_ques_top_words))
        print(np.shape(all_top_data_list_q2_ans_top_words))
        testdata = pd.DataFrame(data)
        self.data = testdata
        self.q2_question = q2_question
        self.q2_answer = q2_answer

        print(testdata['q2'])
        print(testdata['q2_ques_top'])

        x_s1 = np.array(list(self.vocab_processor.fit_transform(testdata['q2'])))
        x_s2 = np.array(list(self.vocab_processor.fit_transform(testdata['q2_ques_top'])))
        self.x = np.concatenate((x_s1, x_s2), axis=1)
        self.y = np.zeros([self.x.shape[0], 2], )
    def train(self,question,answer):
        self.question = question
        self.all_top_data = answer
        self.prepare_data()
        with tf.Session() as sess:
            self.new_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.latest_checkpoint))

            graph = tf.get_default_graph()
            feed_dict = {
                'input_x:0': self.x,
                'input_y:0': self.y,
                'dropout_keep_prob:0': 1.0
            }
            rescorse = graph.get_tensor_by_name("output/scores:0")
            rescorse_ans = sess.run(rescorse, feed_dict)

            rescorse_softmax = tf.nn.softmax(rescorse_ans, 1)
            # print(rescorse_softmax)
            result = rescorse_softmax.eval()
            # print(result)
            data_result = {'q2_ques_top': self.q2_question, 'q2_ans_top': self.q2_answer, "score": result[:, 1]}
            data_result_dataframe = pd.DataFrame(data_result)
            sorded_data_result_dataframe = data_result_dataframe.sort_values(by='score', ascending=False)

            the_final_res = ""
            for index, row in sorded_data_result_dataframe.iterrows():
                the_final_res = the_final_res + row['q2_ques_top'] + "_" + row['q2_ans_top'] +"\n"
            return the_final_res


if __name__ == "__main__":
     sort_by_cnn_func()