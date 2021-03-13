# [실습]
# palceholder 사용

import tensorflow as tf
tf.set_random_seed(66)  # 이걸 사용하지 않으면 돌릴 때 마다 값이 달라진다.

# x_train = [1,2,3]
# y_train = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=None)
y_train = tf.placeholder(tf.float32, shape=None)


W = tf.Variable(tf.random_normal([1]), name = 'weight') # weight값 정규분포에 의한 랜덤한 값 한 개를 집어 넣는다.
b = tf.Variable(tf.random_normal([1]), name = 'bias')


hypothesis = x_train * W + b # y=wx+b의 형태

loss= cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # loss = mse

# optimizer =tf.train.GradientDescentOptimizer(learning_rate=0.01) #옵티마이저
train = tf.train.GradientDescentOptimizer(learning_rate=0.17413885).minimize(cost)



with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(50):
        # 최종 결과 세션 돌릴때 피드딕트 돌리면 된다!
        _, cost_val, W_val, b_val = sess.run([train, loss, W, b], feed_dict={x_train:[1,2,3], y_train : [3,5,7]})
        if step %10 == 0:
            print(step, cost_val, W_val, b_val)
    
    # predict
    print('[4] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_train:[4]}))
    print('[5, 6] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_train:[5,6]}))
    print('[6, 7, 8] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_train:[6,7,8]}))

# 왜 4 / 5,6 / 6,7,8 순으로 예측한지 알아보기!