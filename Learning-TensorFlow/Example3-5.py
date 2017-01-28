import numpy as np
import tensorflow as tf

# === Create data and simulate results =====
x_data = np.random.randn(2000,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2

noise = np.random.randn(1,2000)*0.1
y_data = np.matmul(w_real,x_data.T) + b_real + noise


#######

NUM_STEPS = 30

# === Estimate weights =====
g = tf.Graph()
wb_ = []
with g.as_default():
    x = tf.placeholder('float32',shape=[None,3])
    y_true = tf.placeholder('float32',shape=None)

    w = tf.Variable([[0,0,0]],dtype='float32')
    b = tf.Variable(0,dtype='float32')
    y_pred = tf.matmul(w,tf.transpose(x)) + b 
    loss = tf.reduce_mean(tf.square(y_true-y_pred))

    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    # Before starting, initialize the variables.  We will 'run' this first.
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)      
        for step in range(30):
            sess.run(train,{x: x_data, y_true: y_data})
            if step % 5 == 0:
                print(step, sess.run([w,b]))
                wb_.append(sess.run([w,b]))
