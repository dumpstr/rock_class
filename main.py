from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
# import numpy as np
import matplotlib.pyplot as plt

_URL = 'https://drive.google.com/uc?export=download&confirm=lyck&id=1z1huvTLVtfoRlDKbJmIDm1RibJs-YTOe'
path_to_zip = tf.keras.utils.get_file('geological_similarity.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip),'geological_similarity')

train_dir = os.path.join(PATH, 'train')
test_dir = os.path.join(PATH, 'test')

# define classes
class_names = ['andesite', 'gneiss', 'marble', 'quartzite', 'rhyolite', 'schist']

num_train_examples = sum([len(files) for r, d, files in os.walk(train_dir)])
num_test_examples = sum([len(files) for r, d, files in os.walk(test_dir)])
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

batch_size = 128
epochs = 16
IMG_HEIGHT = 28
IMG_WIDTH = 28

train_img_generator = ImageDataGenerator(horizontal_flip=True,
                                         rotation_range=45,
                                         width_shift_range=.15,
                                         height_shift_range=.15,
                                         zoom_range=0.5) #rescale=1./255,
test_img_generator = ImageDataGenerator() #rescale=1./255

train_data_gen = train_img_generator.flow_from_directory(batch_size=batch_size,
                                                         directory=train_dir,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode='sparse')

test_data_gen = test_img_generator.flow_from_directory(batch_size=batch_size,
                                                     directory=test_dir,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='sparse')

model = tf.keras.Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    #Dropout(0.1),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    #Dropout(0.1),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(batch_size)
#test_dataset = test_dataset.cache().batch(batch_size)

history = model.fit(
    x=train_data_gen,
    epochs=epochs,
    validation_data=test_data_gen
    #validation_steps=9
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()