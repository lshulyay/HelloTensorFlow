# Example instructing reader to create graph based on provided visual representation.

import tensorflow as tf 

a = tf.constant(5.0,name="a")
b = tf.constant(10.0,name="b")

c = tf.mul(a,b,name="c")
d = tf.sin(c,name="d")

e = tf.div(b,d,name="e")

with tf.Session() as sess:
    result = sess.run(e)
    summary_writer = tf.summary.FileWriter("/tmp/tensorflow/graph-b", sess.graph)

print("Result: {}".format(result))