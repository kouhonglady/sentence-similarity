import tensorflow as tf

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # W 形状是[vocab_size, embedding_size] 初始值是从-1到1，均匀分布随机取值
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
            # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
            # tf.nn.embedding_lookup（tensor, id）:tensor就是输入张量，id就是张量对应的索引
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        # 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），
        # enumerate将其组成一个索引序列，利用它可以同时获得索引和值
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    # 指需要做卷积的输入图像
                    self.embedded_chars_expanded,
                    # 相当于CNN中的卷积核
                    W,
                    # 卷积时在图像每一维的步长，这是一个一维的向量，长度4
                    strides=[1, 1, 1, 1],
                    # string类型的量，只能是"SAME","VALID"其中之一，卷积类型
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                # relu函数是将大于0的数保持不变，小于0的数置为0
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        # tf.nn.dropout是TensorFlow里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层。
        # Dropout就是在不同的训练过程中随机扔掉一部分神经元。
        # 也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
        # 但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了。
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                #tf.contrib.layers.xavier_initializer()初始化W
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # tf.nn.l2_loss形如1/2Σw2,一般用于优化的目标函数中的正则项，防止参数太多复杂容易过拟合。
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # tf.argmax就是返回最大的那个数值所在的下标。1 表示 行比较，0表示列比较
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        from sklearn import metrics
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            actuals = tf.argmax(self.input_y, 1)
            ones_like_actuals = tf.ones_like(actuals)
            zeros_like_actuals = tf.zeros_like(actuals)
            tp_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, ones_like_actuals),
                        tf.equal(self.predictions, ones_like_actuals)
                    ),
                    "float"
                )
            )
            tn_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, zeros_like_actuals),
                        tf.equal(self.predictions, zeros_like_actuals)
                    ),
                    "float"
                )
            )

            fp_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, zeros_like_actuals),
                        tf.equal(self.predictions, ones_like_actuals)
                    ),
                    "float"
                )
            )

            fn_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, ones_like_actuals),
                        tf.equal(self.predictions, zeros_like_actuals)
                    ),
                    "float"
                )
            )
            self.precision = tp_op/(tp_op + fp_op)
            self.recall = tp_op/(tp_op + fn_op)
            self.f1_score = (2 * (self.precision * self.recall)) / (self.precision + self.recall)
