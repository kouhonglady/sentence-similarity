import tensorflow as tf
 
a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
 
c = tf.multiply(a, b, name="mul_c")
d = tf.add(a, b, name="add_d")
 
e = tf.add(c, d, name="add_e")
 
sess = tf.Session()
# sess.run(e)
output = sess.run(e)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('tmp', sess.graph)
 
writer.close()
sess.close()



python C:\Users\Admin\AppData\Roaming\Python\Python36\site-packages\tensorboard\main.py --logdir =D:\bigDataCompetition\bank\demo\my_cnn_demo\example