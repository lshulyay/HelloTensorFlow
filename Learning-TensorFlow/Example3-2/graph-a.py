# Example instructing reader to create graph based on provided visual representation.
# Coincidentally also learned how to start TensorBoard in this example 
# to test the graph visualization (making sure it matches the instruction).

import tensorflow as tf  

a = tf.constant(5.0,name="a")
b = tf.constant(10.0,name="b")

c = tf.mul(a,b,name="c") 
d = tf.add(a,b,name="d")

e = tf.sub(c,d,name="e")
f = tf.add(d,c,name="f")

g = tf.div(e,f,name="g")

with tf.Session() as sess:
    result = sess.run(g)
    summary_writer = tf.summary.FileWriter("/tmp/tensorflow/graph-a", sess.graph)

print("Result = {}".format(result))

