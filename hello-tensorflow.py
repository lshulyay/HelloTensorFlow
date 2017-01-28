import tensorflow as tf

h = tf.constant("Hello, ")
t = tf.constant("TensorFlow!")
ht = h + t

with tf.Session() as sess:
    ans = sess.run(ht)

print(ans)