import tensorflow as tf
import numpy as np

a = np.array([0.99, 0.01, 0.99, 0])
a = tf.constant(a)
a = tf.round(a)

b = np.array([0.09, 0.91, 0.99, 0.1])
b = tf.constant(b)
b = tf.round(b)

c = a + b

c = tf.math.floormod(c,2)


tf.print(c)
