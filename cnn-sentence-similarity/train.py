#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import text_cnn
from tensorflow.contrib import learn



# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_path", "./data/original/test_new_train.xlsx", "Data source for the train data.")
tf.flags.DEFINE_string("test_data_path", "./data/original/test_data.csv", "Data source for the test data.")
tf.flags.DEFINE_string("train_data_s1_path", "./data/processed/train_data_s1_path.txt", "S1_to_words of train data.")
tf.flags.DEFINE_string("train_data_s2_path", "./data/processed/train_data_s2_path.txt", "S2_to_words of train data.")
tf.flags.DEFINE_string("word2vec_output_model", "./data/processed/word2vec_output_model.model", "Word2vec_output_model.")
tf.flags.DEFINE_string("word2vec_output_vec", "./data/processed/word2vec_output_vec.vector", "Word2vec_output_vec.")
# tf.flags.DEFINE_string("vocab_processor", "./data/processed/vocab_processor.pickle", "vocab_processor.")
tf.flags.DEFINE_string("vocab_processor", "./data/processed/vocab_processor.txt", "vocab_processor.")




# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")



def preprocess():
    # Data Preparation
    # ==================================================

    # Cut Words
    print("Cutting words...")
    data_helpers.sentence_to_word(FLAGS.train_data_path,FLAGS.train_data_s1_path,FLAGS.train_data_s2_path)

    # Load data
    print("Loading data...")
    x_text_s1,x_text_s2, y = data_helpers.load_data_and_labels(FLAGS.train_data_s1_path, FLAGS.train_data_s2_path,
                                                               FLAGS.train_data_path)

    # x_text_s1, x_text_s2, y = data_helpers.load_data_and_labels("./data/processed/train_data_s1_path.txt","./data/processed/train_data_s2_path.txt",
    #
    #                                                             "./data/original/train_data.csv")
    #s1 = [len(x.split(" ")) for x in x_text_s1]
    # s2=[len(x.split(" "))for x in x_text_s2]
    # s = s1 + s2
    # def drawHist(heights):
    #     pyplot.hist(heights, 100)
    #     pyplot.xlabel('x')
    #     pyplot.ylabel('frequency')
    #     pyplot.title("ssss")
    #     pyplot.show()

    print(len(x_text_s1))
    print(len(x_text_s2))
    print(len(y))


    # Build vocabulary
    # max_s1_length = max([len(x.split(" ")) for x in x_text_s1])
    # max_s2_length = max([len(x.split(" "))for x in x_text_s2])
    # max_document_length = max(max_s1_length,max_s2_length)
    max_document_length = 40

    print("the max document_length is ",max_document_length)

    #让每个句子长度一样 ，短的用0填充
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x_s1 = np.array(list(vocab_processor.fit_transform(x_text_s1)))
    x_s2 = np.array(list(vocab_processor.fit_transform(x_text_s2)))
    x = np.concatenate((x_s1, x_s2), axis=1)

    # 保存和加载词汇表
    vocab_processor.save(FLAGS.vocab_processor)  # 保存


    # vocab = vocab_processor.restore('vocab.pickle')  # 加载

    # Randomly shuffle data
    # 函数shuffle与permutation都是对原来的数组进行重新洗牌（即随机打乱原来的元素顺序）；
    # 区别在于shuffle直接在原来的数组上进行操作，改变原来数组的顺序，无返回值。
    # 而permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组。
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x_s1,x_s2,x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev


def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        # 如果手动设置的设备不存在或者不可用，就会导致tf程序等待或异常，为了防止这种情况，
        # 可以设置tf.ConfigProto()中参数allow_soft_placement=True，允许tf自动选择一个存在并且可用的设备来运行操作。
        # 设置tf.ConfigProto()中参数log_device_placement = True ,可以获取到 operations 和 Tensor 被指派到哪个设备(几号CPU或几号GPU)上运行,
        # 会在终端打印出各项操作是在哪个设备上运行的。

        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            print(y_train.shape)
            cnn = text_cnn.TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            prc_summary = tf.summary.scalar("precision",cnn.precision)
            recall_summary = tf.summary.scalar("recall",cnn.recall)
            f1_summary = tf.summary.scalar("f1_score",cnn.f1_score)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, recall_summary,f1_summary,prc_summary,grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary,recall_summary,f1_summary,prc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")


            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            # 保存二进制模型


            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy,precision,recall,f1_score = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy,cnn.precision,cnn.recall,cnn.f1_score],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g},prc {:g},rcl {:g},f1 {:g}".format(time_str, step, loss, accuracy,precision,recall,f1_score))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy,precision,recall,f1_score= sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy,cnn.precision,cnn.recall,cnn.f1_score],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g},prc {:g},rcl {:g},f1 {:g}".format(time_str, step, loss, accuracy,precision, recall,f1_score))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if current_step %FLAGS.checkpoint_every   == 0:
                    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                                    output_node_names=['output/scores'])
                    pb_file = checkpoint_dir + '/cnn-sentence-similarity'+'.pb'
                    if os.path.isfile(pb_file):
                        os.remove(pb_file)  # 删除文件
                    with tf.gfile.FastGFile(pb_file, mode='wb') as f:
                        f.write(output_graph_def.SerializeToString())
                        print("Saved model  to {}\n".format(pb_file))


def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()