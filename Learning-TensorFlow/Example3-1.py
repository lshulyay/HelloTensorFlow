import tensorflow as tf 

a = tf.constant(5) 
b = tf.constant(2)
c = tf.constant(3)

d = tf.mul(a,b) 
e = tf.add(c,b) 

f = tf.sub(d,e) 

sess = tf.Session() 
outs = sess.run(f) 
sess.close() 

print("outs = {}".format(outs))
