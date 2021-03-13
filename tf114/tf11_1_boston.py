# [실습] 만들거라!!
# 최종 sklearn 의 R2값으로 결론낼것!

from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.compat.v1.set_random_seed(42)

dataset = load_boston()
# print(dataset.data.shape) (442, 10)
x_data = dataset.data
y_data = dataset.target.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8, random_state= 66)

x = tf.placeholder(tf.float32, shape = [None, 13])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random.normal([13,1]), name = 'weight')
b = tf.Variable(tf.random.normal([1]), name = 'bias')

hypothesis = tf.matmul(x, w) + b

loss = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.AdamOptimizer(learning_rate= 0.01).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(25001):
        _, train_loss = sess.run([train, loss], feed_dict= {x:x_train, y:y_train})
        if epoch%100 == 0:
            print(f'Epoch {epoch} === Loss {train_loss}')
    y_pred = sess.run(hypothesis, feed_dict={x:x_test})
    print('R2 : ', r2_score(y_test, y_pred))
