# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
#from IPython.display import display, Image
from scipy import ndimage
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

train_datasets = maybe_pickle(train_folders, 2058) #K is lacking
test_datasets = maybe_pickle(test_folders, 415) #K is lacking
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
            
            
train_size = 28812 ## lowered from 200000
valid_size = 0  ## lowered from 10k   ...not using hyperparameters                 ### enable this for hyperparamter usage
test_size = 5810  ## lowered from 10k

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
"""
#################MORE PICKLE STUFF...probs unnecessary
pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    #'valid_dataset': valid_dataset,                                  ### enable this for hyperparamter usage
    #'#valid_labels': valid_labels,                            ### enable this for hyperparamter usage
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

# Getting statistics of a file using os.stat(file_name)
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
"""
# CHECK FOR OVERLAPS:
"""
def check_overlaps(images1, images2):
    images1.flags.writeable=False
    images2.flags.writeable=False
    start = time.clock()
    hash1 = set([hash(image1.tobytes()) for image1 in images1])
    hash2 = set([hash(image2.tobytes()) for image2 in images2])
    all_overlaps = set.intersection(hash1, hash2)
    return all_overlaps, time.clock()-start

r, execTime = check_overlaps(train_dataset, test_dataset)    
print('Number of overlaps between training and test sets: {}. Execution time: {}.'.format(len(r), execTime))

r, execTime = check_overlaps(train_dataset, valid_dataset)   
print('Number of overlaps between training and validation sets: {}. Execution time: {}.'.format(len(r), execTime))

r, execTime = check_overlaps(valid_dataset, test_dataset) 
print('Number of overlaps between validation and test sets: {}. Execution time: {}.'.format(len(r), execTime))

"""
################################# DO some basic training here....#linear regreression################################
# Here you have ~28000 samples
# 128 x 128 features
# We have to reshape them because scikit-learn expects (n_samples, n_features)
print(train_dataset.shape)
print(test_dataset.shape)

# Prepare training data
samples, width, height = train_dataset.shape
X_train = np.reshape(train_dataset,(samples,width*height))
y_train = train_labels

# Prepare testing data
samples, width, height = test_dataset.shape
X_test = np.reshape(test_dataset,(samples,width*height))
y_test = test_labels

# Import LOGISTIC REGRESSION.....actual stuff done here with this library lol
from sklearn.linear_model import LogisticRegression
print ('initalizating logistic regression...')
# Instantiate
lg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, verbose=0, max_iter=10, n_jobs=-1)
print ('beginning fitting')
# Fit
lg.fit(X_train, y_train)
print ('done fitting')
# Predict
y_pred = lg.predict(X_test)
print ('dont predicting')
# Score
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print ('accuracy: ', accuracy)
