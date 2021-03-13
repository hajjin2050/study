import tensorflow as tf
tf.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name='weight')

print(W)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(W)
print(aaa)
sess.close()

sess = tf.InteractiveSession()
sess = tf.compat.v1.interactiveSession()
# sess.run(tf.global_variables_initializer())
sess.run(tf.caompat.v1.global_variables_initializer())
bbb = W.eval() #  변수 . eval
print("bbb:", bbb)
sess.close()

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializerl())
ccc = W.eval(session = sess)
print("ccc:", ccc)
sess.close()