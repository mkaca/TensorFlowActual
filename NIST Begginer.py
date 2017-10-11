#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#nist = input_data.read_data_sets("NIST_data/") #training data
#128x128
import tensorflow as tf
from random import randint
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from numpy import array
import scipy.misc
import os
import time
import webbrowser

num_classes = 12
#webbrowser.open("C:\\Users\\dabes\\Desktop\\TensorFlow\\NIST_data\\4b\\hsf_4\\hsf_4_00000.png")

fn = os.listdir("C:\\Users\\dabes\\Desktop\\TensorFlow\\NIST_data\\4b\\hsf_4\\")
print(fn)

# Display first 20 images 
#for file in fn[:2]:
    #path = "C:\\Users\\dabes\\Desktop\\TensorFlow\\NIST_data\\4b\\hsf_4\\" + file
    #webbrowser.open(path)
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset

#PICKLING CONVERTS OBJECTS TO BYTES        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)   


"""

### RESHAPE IMAGES HERE
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 14]))
b = tf.Variable(tf.zeros([14]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 14])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):  # reduced from 20000 to 1000 so it's 20x faster
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
prediction=tf.argmax(y,1)

num = randint(0, mnist.test.images.shape[0])
img = mnist.test.images[num]
print ((img))
print ("    ")
print ("    ")

imagePath='C:\\Users\\dabes\Desktop\\TensorFlow\\TensorFlow#3TestImage.png'
#x_data = cv2.imread(imagePath)
img = Image.open(imagePath).convert('F')
img2 = numpy.array(img)
x_data = scipy.misc.imresize(img2, (28,28))
x_data = x_data.flatten()

print ("predictions", prediction.eval(feed_dict={x: [x_data]}, session=sess))
"""
