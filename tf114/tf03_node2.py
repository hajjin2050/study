#뺄셈 곱셈 나누셈 맹그러라!


import tensorflow as tf

#곱하기
# node1 = tf.constant(2.0)
# node2 = tf.constant(3.0)
# node3 = tf.matmul(node1, node2)
sess = tf.compat.v1.Session()
#뺼셈
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)
node3 = tf.subtract(node1, node2)

#나눗셈
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)
node3 = tf.truediv(node1, node2)

#나머지 출력
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)
node3 = tf.math.mod(nod1, node2)