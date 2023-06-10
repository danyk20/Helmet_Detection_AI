import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='reflect')
