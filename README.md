# Requirements

- Pillow
- SciPy
- Tensorflow

# Main dataset

- Kaggle https://www.kaggle.com/datasets/meliodassourav/traffic-violation-dataset-v3?resource=download
- Many problem:
    - duplicate images
    - incorrect labels
    - drawing or computer animation instead of pictures
    - watermarks
    - not uniform size/quality
    - blured
    - small

# Data augmentation

- using ImageDataGenerator from keras_preprocessing.image
    - increase number of training images by:
        - rotation by up to 15 degrees (real photo conditions)
        - zoom in up to 20%
        - mirror effect (horizontal flip)
        - add missing pixel when rotation or moving by reflecting existing in the original picture
        - move horizontally and vertically up to 15 %
        - shear transformation up to 20% https://en.wikipedia.org/wiki/Shear_mapping

# Optimizer Adam

- binary_crossentropy as loss since we have only 2 categories (helmet yes/no)
- learning_rate is 0.005 because we are going to train model often during development 
  - we rather sacrifice minor accuracy improvement but gain faster trained model

# Convolution layer
- tf.keras.layers.Conv2D(64, kernel_size=11, strides=8, activation='relu', input_shape=(64, 64, 3))
  - 64 filters
  - filter is 11x11 pixel big
  - each filter center is 8 pixels away (it reduces computational time)
  - input image are 64x64 pixels with rgb (3) colors

# Pooling layer
- to reduce number of parameter in the model -> faster training
- tf.keras.layers.MaxPooling2D(2, strides=2)
  - 2 size of the pooling window
    - this will return from 4 pixels (2x2) only one with the max value


# Dropout
- In each step we ignore some units
- Strategy to avoid over-fitting
- Slow down learning -> make it more incrementally careful
- tf.keras.layers.Dropout(0.5)
  - we are ignoring every other unit

# Batch normalisation
- Rescale output to have sample mean and standard deviation of 1
- Make learning faster
- tf.keras.layers.BatchNormalization()