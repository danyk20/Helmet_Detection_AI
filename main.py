import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array

train_data_generator = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='reflect')

pic = load_img('test_data/520.jpg')
pic_array = img_to_array(pic)
pic_array = pic_array.reshape((1,) + pic_array.shape)

count = 0
for batch in train_data_generator.flow(pic_array, batch_size=1, save_to_dir='test_data/generated_images', save_prefix='generated', save_format='jpeg'):
    count += 1
    if count == 4:
        break

