from tensorflow.python.framework.ops import disable_eager_execution

import tensorflow as tf

print(tf.executing_eagerly()) #Flase

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())



print(tf.__version__)

hello = tf.constant("Hello World")
print(hello)
#Tensor("Cons:0",vshape=(),dtype=string)

sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello))
