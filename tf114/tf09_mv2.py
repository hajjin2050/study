import tensorflow as tf
tf.set_random_seed(66)

x_data = [[73, 51, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]

y_data = [[152],
          [185],
          [180],
          [205],
          [142]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([3,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b


cost = tf.reduce_mean(tf.square(hypothesis - y)) # loss='mse'

train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost) # optimizer + train
# train = tf.train.GradientDescentOptimizer(learning_rate=0.17413885).minimize(cost) # optimizer + train



# with문 사용해서 자동으로 sess가 닫히도록 할수도 있다.
with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(5001):
        cost_val, w_val, b_val, hy_val, _ =sess.run([cost,w,b,hypothesis,train], feed_dict={x:x_data,y:y_data})
        if step %20 == 0:
            print(step, cost_val, w_val, b_val) # epoch, loss, weight, bias
            print(step, "cost :", cost_val, "\n", hy_val)