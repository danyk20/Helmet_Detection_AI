from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import keras.models as models
from keras.optimizers.legacy import Adam
import numpy as np

from keras.preprocessing import image

TEST_IMAGE_PATH = "/Users/danielkosc/Documents/MUNI/Spring2023/ML/project/data_copy/Final_data/nohelmet_c/generated_0_52.jpg"

# model configuration
IMAGE_SIZE = 128
IMAGE_COLORS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 20
EPOCHS = 100
VALIDATION_SPLIT = 0.2
METRIC = 'accuracy'
LOSS_FUNCTION = 'binary_crossentropy'

# image generation
RESCALE = 1
SHUFFLE = True
CLASS_MODE = 'binary'
DATA_PATH = '/Users/danielkosc/Documents/MUNI/Spring2023/ML/project/data_copy/Final_data'

# augmentation configuration
MULTIPLIER = 4
ROTATION = 35
ZOOM = 0.15
SHEAR = 0.2
HEIGHT_SHIFT = 0.05
WIDTH_SHIFT = 0.15
FILL_MODE = 'reflect'
HORIZONTAL_FLIP = True
GENERATED_DIR_PATH = '/generated_images'
PHOTO_PREFIX = 'generated'
INPUT_FORMAT = 'png'
OUTPUT_FORMAT = 'jpg'


def augment_pictures(multiplier, path):
    train_data_generator = ImageDataGenerator(rotation_range=ROTATION,
                                              width_shift_range=WIDTH_SHIFT,
                                              height_shift_range=HEIGHT_SHIFT,
                                              shear_range=SHEAR,
                                              zoom_range=ZOOM,
                                              horizontal_flip=HORIZONTAL_FLIP,
                                              fill_mode=FILL_MODE
                                              )
    pictures = [f for f in listdir(path) if isfile(join(path, f)) and f.split(".")[-1] == INPUT_FORMAT]
    for picture in pictures:
        pic = load_img(path + '/' + picture)
        pic_array = img_to_array(pic)
        pic_array = pic_array.reshape((1,) + pic_array.shape)

        count = 0
        for _ in train_data_generator.flow(pic_array, batch_size=BATCH_SIZE, save_to_dir=path + GENERATED_DIR_PATH,
                                           save_prefix=PHOTO_PREFIX, save_format=OUTPUT_FORMAT):
            count += 1
            if count == multiplier:
                break


def get_model():
    result = Sequential()
    result.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_COLORS)))
    result.add(MaxPooling2D((2, 2)))
    result.add(Conv2D(64, (3, 3), activation='relu'))
    result.add(MaxPooling2D((2, 2)))
    result.add(Conv2D(128, (3, 3), activation='relu'))
    result.add(MaxPooling2D((2, 2)))
    result.add(Flatten())
    result.add(Dense(128, activation='relu'))
    result.add(Dense(1, activation='sigmoid'))
    # get description
    result.summary()
    return result


def compile_model(raw_model):
    raw_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=LOSS_FUNCTION,
        metrics=[METRIC]
    )


def fit_model(empty_model, train_generator, validation_generator):
    result = empty_model.fit(train_generator,
                             validation_split=VALIDATION_SPLIT,
                             steps_per_epoch=train_generator.samples // BATCH_SIZE,
                             validation_data=validation_generator,
                             validation_steps=validation_generator.samples // BATCH_SIZE,
                             epochs=EPOCHS)
    return result


def plot_graph(hist):
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


def demo():
    model = models.load_model("trained_model.h5")
    # predicting images
    path = TEST_IMAGE_PATH
    img = image.load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=1)
    print(classes[0])
    if classes[0] > 0.5:
        print("There is driver without helmet!")
    else:
        print("There is driver with helmet!")
    plt.imshow(img)


def save_model():
    augment_pictures(MULTIPLIER, 'data_copy/nohelmet_b')
    model = get_model()
    compile_model(model)
    datagen = ImageDataGenerator(rescale=RESCALE, validation_split=VALIDATION_SPLIT)
    train_generator = datagen.flow_from_directory(
        directory=DATA_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        shuffle=SHUFFLE,
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        directory=DATA_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        shuffle=SHUFFLE,
        subset='validation'
    )
    hist = fit_model(model, train_generator, validation_generator)
    model.save("trained_model.h5", include_optimizer=True)
    plot_graph(hist)


demo()
