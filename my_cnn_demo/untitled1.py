# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 00:00:53 2018

@author: Admin
"""

import tensorflow as tf
with tf.name_scope('graph') as scope:
     matrix1 = tf.constant([[3., 3.]],name ='matrix1')  #1 row by 2 column
     matrix2 = tf.constant([[2.],[2.]],name ='matrix2') # 2 row by 1 column
     product = tf.matmul(matrix1, matrix2,name='product')
sess = tf.Session()
writer = tf.summary.FileWriter("./test", sess.graph)
init = tf.global_variables_initializer()
sess.run(init)