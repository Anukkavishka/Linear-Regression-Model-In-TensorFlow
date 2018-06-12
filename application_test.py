#import tensorflow

import tensorflow as tf
#to start a tensorflow session
sess=tf.Session()

#verfiy we can print a string

hello=tf.constant("Welcome to TensorFlow")
print(sess.run(hello))
#testing simple 
a=tf.constant(10)
b=tf.constant(20)

print('a+b = {0}'.format(sess.run(a+b)))