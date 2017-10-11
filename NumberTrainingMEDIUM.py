# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
#from IPython.display import display, Image
from scipy import ndimage, misc
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import webbrowser
from PIL import Image 
import time



# name all the paths here
test_folders = ['NIST_data/0/hsf_0' , 'NIST_data/1/hsf_0', 'NIST_data/2/hsf_0' , 'NIST_data/3/hsf_0', 'NIST_data/4/hsf_0' , 'NIST_data/5/hsf_0',
                'NIST_data/6/hsf_0' , 'NIST_data/7/hsf_0', 'NIST_data/8/hsf_0' , 'NIST_data/9/hsf_0', 'NIST_data/A/hsf_0' , 'NIST_data/J/hsf_0',
                'NIST_data/K/hsf_0' , 'NIST_data/Q/hsf_0']
train_folders = ['NIST_data/0/hsf_1' , 'NIST_data/1/hsf_1', 'NIST_data/2/hsf_1' , 'NIST_data/3/hsf_1', 'NIST_data/4/hsf_1' , 'NIST_data/5/hsf_1',
                'NIST_data/6/hsf_1' , 'NIST_data/7/hsf_1', 'NIST_data/8/hsf_1' , 'NIST_data/9/hsf_1', 'NIST_data/A/hsf_1' , 'NIST_data/J/hsf_1',
                'NIST_data/K/hsf_1' , 'NIST_data/Q/hsf_1']

#webbrowser.open('C:\\Users\\dabes\\Desktop\\TensorFlow\\NIST_data\\1\\hsf_0\\hsf_0_00000.png')

#fn = os.listdir('NIST_data/2/hsf_0')
#print(fn)

# Display first 20 images 
#for file in fn[:4]:
    #path = 'C:\\Users\\dabes\\Desktop\\TensorFlow\\NIST_data\\1\\hsf_0\\' + file
    #webbrowser.open(path)

image_size = 128  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  #print (len(dataset))
  #print (len(dataset[0]))
  #print (len(dataset[0][0]))
  print(folder)
  num_images = 0
  for image in image_files:
    
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file, mode='L').astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        print ('image_data.shape: ',image_data.shape)
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      #print (len(image_data))
      #print (len(image_data[0]))
      dataset[num_images, :, :] = image_data

      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
      print (num_images)
  
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
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

train_datasets = maybe_pickle(train_folders, 801) #K is lacking
test_datasets = maybe_pickle(test_folders, 101) #K is lacking
"""
###### IMPORT PICKLE RICKKKKKKK
test_values = ['test value' , 'est value2', 'potato']
file_Name = "testfile"
# Open the file for writing
fileObject = open(file_Name,'wb') 

# This writes the object a to the
# file named 'testfile'
pickle.dump(test_values, fileObject)   

# Then we close the fileObject
fileObject.close()

# We then open the file for reading
fileObject = open(file_Name,'rb')

# And the object from the file into var b
test_values_loaded = pickle.load(fileObject) 
#display((test_values_loaded))
#display(test_values == test_values_loaded)

############# NOW WE VERIFY IF DATA STILL LOOKS GOOD

###IMPORTANT:::::::::::: MUST CALL THIS FUNCTION TO DISPLAY MATPLOT IMAGE IN JUPYTER 
#%matplotlib inline
###IMPORTANT:::::::::::: MUST CALL THIS FUNCTION TO DISPLAY MATPLOT IMAGE IN JUPYTER 

# index 0 should be all As, 1 = all Bs, etc.
pickle_file = train_datasets[0]  

# With would automatically close the file after the nested block of code
with open(pickle_file, 'rb') as f:
    
    # unpickle
    letter_set = pickle.load(f)  
    
    # pick a random image index
    sample_idx = np.random.randint(len(letter_set))
    
    # extract a 2D slice
    sample_image = letter_set[sample_idx, :, :]  
    plt.figure()
    
    # display it
    plt.imshow(sample_image)
"""
#=========================================================================
### Make Arrays ###
    
def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      print (pickle_file)
      print ('label: ', label)
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                   

        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 2100 ## lowered from 200000 ###was 4200
valid_size = 0  ## lowered from 10k   ...not using hyperparameters                 ### enable this for hyperparamter usage
test_size = 1400  ## lowered from 10k

_, _, train_dataset, train_labels = merge_datasets(   ### Change the first 2 params to valid data and valid label for hyperparam usage
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)    

print('Training:', train_dataset.shape, train_labels.shape)
#print('Validation:', valid_dataset.shape, valid_labels.shape)                ### enable this for hyperparamter usage
print('Testing:', test_dataset.shape, test_labels.shape) 

##### SHUFFLE DATA TO MAKE SURE IT'S ALL GOOD
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
#valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)                    ### enable this for hyperparamter usage


# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(train_dataset,train_labels))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(train_dataset)
print ('number of samples',n_samples)
data = train_dataset.reshape((n_samples, -1))
print ('datalen1: ',len(data))
print ('datalen2: ',len(data[0]))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.002)

print ('starting fit')
# We learn the digits on the first half of the digits
classifier.fit(data, train_labels)
print('starting predict')
# Now predict the value of the test dataset
expected = test_labels
predicted = classifier.predict(test_dataset.reshape(len(test_dataset),-1))

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))  #expected size has to be the same as predicted size!!!!!!!!
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(test_dataset, predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()

imagePath='C:\\Users\\dabes\Desktop\\TensorFlow\\TensorFlow#4TestImage.png'
imgOrg = Image.open(imagePath).convert('1')
img = np.array(imgOrg)
print (img)

img = misc.imresize(img,(128,128))
img = img.flatten()
prediction2 = classifier.predict([img])
print ('pred2', prediction2)
plt.axis('off')
plt.imshow(imgOrg, cmap=plt.cm.gray_r, interpolation='nearest')
plt.title('PredictionTESTTTT: %i' % prediction2)
plt.show()
plt.imshow(img,cmap=plt.cm.gray_r, interpolation ='nearest')
plt.show()
img2= Image.open("C:\\Users\\dabes\\Desktop\\TensorFlow\\NIST_data\\8\\hsf_4\\hsf_4_00001.png").convert('1')
plt.imshow(img2, cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
# i wanna print out the number 4 in the same format as the testing ones, fora comparison
