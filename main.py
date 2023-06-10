import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from os import listdir
from os.path import isfile, join


def augment_pictures(multiplier, path):
    train_data_generator = ImageDataGenerator(rotation_range=15,
                                              width_shift_range=0.15,
                                              height_shift_range=0.15,
                                              shear_range=0.2,
                                              zoom_range=0.2,
                                              horizontal_flip=True,
                                              fill_mode='reflect')
    pictures = [f for f in listdir(path) if isfile(join(path, f)) and f.split(".")[-1] == 'jpg']
    for picture in pictures:
        pic = load_img(path + '/' + picture)
        pic_array = img_to_array(pic)
        pic_array = pic_array.reshape((1,) + pic_array.shape)

        count = 0
        for batch in train_data_generator.flow(pic_array, batch_size=1, save_to_dir=path + '/generated_images',
                                               save_prefix='generated', save_format='jpeg'):
            count += 1
            if count == multiplier:
                break



augment_pictures(4, 'test_data')
