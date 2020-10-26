

import os
import zipfile
import tensorflow

os.chdir('C:\\Users\\User\\Desktop')

local_zip = 'Xrayimage/XrayArchive.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/Xrayimage')
zip_ref.close()


base_dir = '/XrayImage/chest_xray'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test')

# Directory with our training healthy/sick pictures
train_healthy_dir = os.path.join(train_dir, 'NORMAL')
train_sick_dir = os.path.join(train_dir, 'PNEUMONIA')

# Directory with our validation healthy/sick pictures
validation_healthy_dir = os.path.join(validation_dir, 'NORMAL')
validation_sick_dir = os.path.join(validation_dir, 'PNEUMONIA')

train_healthy_fnames = os.listdir( train_healthy_dir )
train_sick_fnames = os.listdir( train_sick_dir )




import tensorflow as tf

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 180x180 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])



from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.



train_datagen = ImageDataGenerator( rescale=1./255,
     rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
     shear_range=0.2,
      zoom_range=0.2,
     fill_mode='nearest')



test_datagen  = ImageDataGenerator( rescale = 1.0/255 )




train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=8,
                                                    class_mode='binary',
                                                    target_size=(180, 180))     


validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=8,
                                                         class_mode  = 'binary',
                                                         target_size = (180, 180))




from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])




history = model.fit(train_generator,
                              validation_data=validation_generator,
                              epochs=25,
                              verbose=1)




import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()



plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation Loss')
plt.legend(loc=0)
plt.figure()


plt.show()


