import tensorflow as tf
import numpy as np
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50



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

train_generator = train_datagen.flow_from_directory(
        batch_size=batch_size,
		directory='../datasets/HAM/train',
        target_size=(600, 450), 
        classes = ['mask','no-mask'],
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        batch_size=batch_size,
        directory='../datasets/HAM/validation',
        target_size=(600, 450), 
        classes = ['mask','no-mask'],
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
		batch_size=batch_size,
        directory='../datasets/HAM/test',
        target_size=(600, 450), 
        classes = ['mask','no-mask'],
        class_mode='categorical')


### Pre-trained Model w/ Transfer Learning

model = tf.keras.applications.ResNet51(include_top=False,weights='imagenet')

# transfer learning
for i in model.layers:
  i.trainable = False

global_avg = tf.keras.layers.GlobalAveragePooling2D()(model.output)
drop_out = tf.keras.layers.Dropout(0.4)(global_avg)
out = tf.keras.layers.Dense(2,activation='sigmoid')(drop_out)
resnet = tf.keras.Model(inputs=[model.input],outputs=[out])