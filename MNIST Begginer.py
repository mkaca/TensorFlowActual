from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
from random import randint
import numpy
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from numpy import array
import scipy.misc
import webbrowser
numpy.set_printoptions(linewidth=2000)


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(4000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

num = randint(0, mnist.test.images.shape[0])
img = mnist.test.images[num]

#print ((img))
print ("    ")
print ("    ")
numpy.set_printoptions(linewidth=500)
#imagePath = 'C:\\Users\\dabes\\Desktop\\TensorFlow\\NIST_data\\0\\hsf_0\\hsf_0_00014.png'
imagePath='C:\\Users\\dabes\Desktop\\TensorFlow\\TensorFlow#10TestImage.png'
#x_data = cv2.imread(imagePath)
img = Image.open(imagePath).convert('F')
print (img)
print ("    ")
img2 = numpy.array(img)
print (img2)
print ("    ")
x_data = scipy.misc.imresize(img2, (28,28))
print ("    ")
print (x_data)
x_data = x_data.flatten()
print ("    ")
print (x_data)
print ("    ")

#img = numpy.reshape(img, (28,28))
#print (img)
#img = img.flatten()
#x_data = img
prediction=tf.argmax(y,1)
print ("predictions", prediction.eval(feed_dict={x: [x_data]}))
