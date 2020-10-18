 

import os
import zipfile

# I have the files saved on my desktop in a folder called Xrayimage, which contains the zip images.
os.chdir('C:\\Users\\User\\Desktop')


# Access the zip objects in the folder I have saved
local_zip = 'Xrayimage/XrayArchive.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/Xrayimage')
zip_ref.close()

# Create base directory
base_dir = '/XrayImage/chest_xray'


# Seperate the training and testing data
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test')

# Directory with our healthy and sick (pneumonia) data
train_healthy_dir = os.path.join(train_dir, 'NORMAL')
train_sick_dir = os.path.join(train_dir, 'PNEUMONIA')

# Directory with our testing healthy (normal) vs sick (pneomonia)
validation_healthy_dir = os.path.join(validation_dir, 'NORMAL')
validation_sick_dir = os.path.join(validation_dir, 'PNEUMONIA')

train_healthy_fnames = os.listdir( train_healthy_dir )
train_sick_fnames = os.listdir( train_sick_dir )

# Check names of images
print(train_healthy_fnames[:15])
print(train_sick_fnames[:15])

# Check how many images are in each directory.
print('total training Healthy images :', len(os.listdir(train_healthy_dir ) ))
print('total training Sick images :', len(os.listdir(train_sick_dir)))

print('total validation Healthy images :', len(os.listdir(validation_healthy_dir)))
print('total validation Sick images :', len(os.listdir(validation_sick_dir)))




# Plot some of the images in 

%matplotlib inline

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

pic_index = 0 # Index for iterating over images


# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_healthy_pix = [os.path.join(train_healthy_dir, fname) 
                for fname in train_healthy_fnames[ pic_index-8:pic_index] 
               ]

next_sick_pix = [os.path.join(train_sick_dir, fname) 
                for fname in train_sick_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_healthy_pix+next_sick_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


import tensorflow as tf

# Create the model, we include multiple convolutional layers, some max pooling and finally a dense layer that connects to an output with sigmoid activation.


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 180x180 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), 
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'), 
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')  
])



# We use RMSprop optimizer with a small learning rate.

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['accuracy'])


# We use ImageDataGenerator to automatically get our images ready for training

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.
# We also include several image augmentation arguments to increase the number of training examples.

train_datagen = ImageDataGenerator( rescale=1./255,
     rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
     shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
     fill_mode='nearest')



test_datagen  = ImageDataGenerator( rescale = 1.0/255. )




train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(180, 180))     



validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=25,
                                                         class_mode  = 'binary',
                                                         target_size = (180, 180))


history = model.fit(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=10,
                              epochs=20,
                              validation_steps=10,
                              verbose=2)


# We can also visualize layers inside the middle of the network.

import numpy as np
import random
from   tensorflow.keras.preprocessing.image import img_to_array, load_img

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]

#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

# Let's prepare a random input image of a cat or dog from the training set.
healthy_img_files = [os.path.join(train_healthy_dir, f) for f in train_healthy_fnames]
sick_img_files = [os.path.join(train_sick_dir, f) for f in train_sick_fnames]

img_path = random.choice(healthy_img_files + sick_img_files)
img = load_img(img_path, target_size=(150, 150))  # this is a PIL image

x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255.0

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# -----------------------------------------------------------------------
# Now let's display our representations
# -----------------------------------------------------------------------
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  
  if len(feature_map.shape) == 4:
    
    #-------------------------------------------
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    #-------------------------------------------
    n_features = feature_map.shape[-1]  # number of features in the feature map
    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
    
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    
    #-------------------------------------------------
    # Postprocess the feature to be visually palatable
    #-------------------------------------------------
    for i in range(n_features):
      x  = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std ()
      x *=  64
      x += 128
      x  = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

    #-----------------
    # Display the grid
    #-----------------

    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 
