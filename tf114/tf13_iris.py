import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



tf.set_random_seed(42)

dataset = load_iris()

x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(-1,1)
# print(x_data.shape, y_data.shape) #(150, 4) (150, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8, random_state=44)
hypothesis = tf.nn.softmax(tf.matmul(x, w)+ b)
#
# loss = tf.reduce_mean(tf.square(hypthesis - y)) #mse
#categorical_corssentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis= 1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict = {x : x_data, y: y_data})
        if step % 200 == 0 :
            print(step, cost_val)

    a = sess.run(hypothesis, feed_dict={x:[[1, 11, 7, 9]]})
    print('Accuracy : ', a)
    y_pred = sess.run(hypothesis, feed_dict={x:x_test})
    y_pred = np.where(y_pred>0.5, 1, 0)
    print('Acc_score : ', accuracy_score(y_test, y_pred))
