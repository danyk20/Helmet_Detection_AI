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
