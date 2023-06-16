import tensorflow as tf
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from os import listdir
from os.path import isfile, join

batch_size = 10
def augment_pictures(multiplier, path):
    train_data_generator = ImageDataGenerator(rotation_range=35,
                                              width_shift_range=0.15,
                                              height_shift_range=0.05,
                                              shear_range=0.2,
                                              zoom_range=0.15,
                                              horizontal_flip=True,
                                              fill_mode='reflect')
    pictures = [f for f in listdir(path) if isfile(join(path, f)) and f.split(".")[-1] == 'png']
    for picture in pictures:
        pic = load_img(path + '/' + picture)
        pic_array = img_to_array(pic)
        pic_array = pic_array.reshape((1,) + pic_array.shape)

        count = 0
        for batch in train_data_generator.flow(pic_array, batch_size=batch_size, save_to_dir=path + '/generated_images',
                                               save_prefix='generated', save_format='jpg'):
            count += 1
            if count == multiplier:
                break


def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # get description
    model.summary()
    return model


def compile_model(model):
    from keras.optimizers.legacy import Adam
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


def fit_model(model, train_generator, validation_generator):
    hist = model.fit(train_generator,
                     validation_split=0.2,
                     steps_per_epoch=train_generator.samples // batch_size,
                     validation_data=validation_generator,
                     validation_steps=validation_generator.samples // batch_size,
                     epochs=10)
    return hist


augment_pictures(4, 'data_copy/nohelmet_b')
model = get_model()
compile_model(model)
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    directory='/Users/danielkosc/Documents/MUNI/Spring2023/ML/project/data_copy/Final_data',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    directory='/Users/danielkosc/Documents/MUNI/Spring2023/ML/project/data_copy/Final_data',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    subset='validation'
)
hist = fit_model(model, train_generator, validation_generator)

import matplotlib.pyplot as plt
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()


import numpy as np

from keras.preprocessing import image
# predicting images
path = "/Users/danielkosc/Documents/MUNI/Spring2023/ML/project/data_copy/Final_data/nohelmet_c/generated_0_52.jpg"
img = image.load_img(path, target_size=(128, 128))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=1)
print(classes[0])
if classes[0]>0.5:
    print("is with helmet")
else:
    print(" is without helmet")
plt.imshow(img)
