import tensorflow as tf
print(tf.__version__)

hello = tf.constant("Hello World")
print(hello)
#Tensor("Cons:0",vshape=(),dtype=string)

sess = tf.Session()
print(sess.run(hello))
