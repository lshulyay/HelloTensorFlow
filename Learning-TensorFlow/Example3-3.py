import tensorflow as tf

with tf.Graph().as_default(): 
    x = tf.random_normal((5,10), mean=0.0, stddev=1.0) 
    w = tf.random_normal((10,1), mean=0.0, stddev=1.0) 
    b = tf.fill((5,1),-1.) 
    xw = tf.matmul(x,w) 

    xwb = xw + b 
    s = tf.sigmoid(xwb) 

    sess = tf.Session()             
    outs = sess.run(s)
    sess.close()

print("outs = {}".format(outs))