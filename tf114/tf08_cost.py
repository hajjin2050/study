import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [3., 5., 7.]

W  = tf.compat.v1.placeholder(tf.float32)

hypothesis = W * x

cost = tf.reduce_mean(tf.square(hypothesis - y))

w_history = []
cost_history = []

with tf.compat.v1.Session() as sess :
    for i in range(-30, 50):
        curr_w = i * 0.1
        curr_cost = sess.run(cost, feed_dict={W:curr_w})

        w_history.append(curr_w)
        cost_history.append(curr_cost)
print("==================================")
print(w_history)
print("==================================")
print(cost_history)

plt.pot(w_history, cost_history)
plt.history()