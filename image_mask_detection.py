import tensorflow as tf
import numpy as np
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
import os

### Organizing classes into training, validation, and test
# print("Organizing datasets")
# masked_imgs = os.listdir('datasets/compiled/with_mask copy')
# no_mask_imgs = os.listdir('datasets/compiled/without_mask copy')

# i = 0
# for img in masked_imgs:
# 	if i < int(0.64*len(masked_imgs)):
# 		os.rename('datasets/compiled/with_mask copy/' + img, 'datasets/compiled/train/mask/' + img)
# 		i += 1
# 	elif i < int(0.8*len(masked_imgs)):
# 		os.rename('datasets/compiled/with_mask copy/' + img, 'datasets/compiled/validation/mask/' + img)
# 		i += 1
# 	else:
# 		os.rename('datasets/compiled/with_mask copy/' + img, 'datasets/compiled/test/mask/' + img)
# 		i += 1

# j = 0
# for img in no_mask_imgs:
# 	if j < int(0.64*len(no_mask_imgs)):
# 		os.rename('datasets/compiled/without_mask copy/' + img,'datasets/compiled/train/no-mask/' + img)
# 		j += 1
# 	elif j < int(0.8*len(no_mask_imgs)):
# 		os.rename('datasets/compiled/without_mask copy/' + img, 'datasets/compiled/validation/no-mask/' + img)
# 		j += 1
# 	else:
# 		os.rename('datasets/compiled/without_mask copy/' + img, 'datasets/compiled/test/no-mask/' + img)
# 		j += 1

# print("Images reorganized")

train_datagen = ImageDataGenerator(
    width_shift_range=0.3,
    height_shift_range=0.3,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255)

validation_datagen = ImageDataGenerator(
    width_shift_range=0.3,
    height_shift_range=0.3,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255)

test_datagen = ImageDataGenerator(
    width_shift_range=0.3,
    height_shift_range=0.3,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255)

epochs = 10
batch_size = 32

train_generator = train_datagen.flow_from_directory(
        batch_size=batch_size,
		directory='datasets/compiled/train/',
        target_size=(200, 300), 
        classes = ['mask','no-mask'],
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        batch_size=batch_size,
        directory='datasets/compiled/validation/',
        target_size=(200, 300), 
        classes = ['mask','no-mask'],
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
		batch_size=batch_size,
        directory='datasets/compiled/test/',
        target_size=(200, 300), 
        classes = ['mask','no-mask'],
        class_mode='categorical')


### Pre-trained Model w/ Transfer Learning

model = tf.keras.applications.ResNet50(include_top=False,weights='imagenet')

# transfer learning
for i in model.layers:
  i.trainable = False

global_avg = tf.keras.layers.GlobalAveragePooling2D()(model.output)
drop_out = tf.keras.layers.Dropout(0.4)(global_avg)
out = tf.keras.layers.Dense(2,activation='sigmoid')(drop_out)
resnet = tf.keras.Model(inputs=[model.input],outputs=[out])

resnet.compile(optimizer=tf.keras.optimizers.Adam(),loss="binary_crossentropy",metrics=["accuracy"])

history = resnet.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

