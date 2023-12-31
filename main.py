import os

from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import keras.models as models
from keras.optimizers.legacy import Adam
import numpy as np
from PIL import Image
from pathlib import Path

from keras.preprocessing import image

from utils import resize_images

# paths
TRAINED_MODEL = 'trained_model.h5'
HELMET_INPUT = 'train/helmet'
HELMET_TRAINING = 'data/helmet'
NO_HELMET_INPUT = 'train/no_helmet'
NO_HELMET_TRAINING = 'data/no_helmet'
TEST_IMAGE_PATH = "test/helmet_0.jpg"
TEST_IMAGE_PATH_2 = "test/no_helmet_0.jpg"
# False negative 2
# False positive 1, 4

# model configuration
IMAGE_SIZE = 96
IMAGE_COLORS = 3
LEARNING_RATE = 0.00005
BATCH_SIZE = 64
EPOCHS = 30
VALIDATION_SPLIT = 0.2
METRIC = 'accuracy'
LOSS_FUNCTION = 'binary_crossentropy'

# image generation
RESCALE = 1
SHUFFLE = True
CLASS_MODE = 'binary'
DATA_PATH = 'data'

# augmentation configuration
MULTIPLIER = 4
ROTATION = 35
ZOOM = 0.1
SHEAR = 0.2
HEIGHT_SHIFT = 0
WIDTH_SHIFT = 0.2
FILL_MODE = 'reflect'
HORIZONTAL_FLIP = True
# GENERATED_DIR_PATH = '/generated_images'
PHOTO_PREFIX = 'generated'
ALLOWED_INPUT_FORMATS = ['png', 'jpg']
OUTPUT_FORMAT = 'jpg'


def augment_pictures(multiplier, path, train_dir_path):
    # Change all pictures to uniform size
    resized_pictures_path: str = '/'.join(path.split('/')[:-1])
    resized_pictures_path += '/' + 'resized_' + path.split('/')[-1]
    resize_images(path, resized_pictures_path, (IMAGE_SIZE, IMAGE_SIZE))

    # augment
    generator = ImageDataGenerator(rotation_range=ROTATION,
                                   width_shift_range=WIDTH_SHIFT,
                                   height_shift_range=HEIGHT_SHIFT,
                                   shear_range=SHEAR,
                                   zoom_range=ZOOM,
                                   horizontal_flip=HORIZONTAL_FLIP,
                                   fill_mode=FILL_MODE
                                   )
    pictures = [f for f in listdir(resized_pictures_path) if
                isfile(join(resized_pictures_path, f)) and f.split(".")[-1] in ALLOWED_INPUT_FORMATS]
    Path(train_dir_path).mkdir(parents=True, exist_ok=True)
    generated_pictures = 0
    for picture in pictures:
        print('generated pictures: ' + str(generated_pictures) + '/' + str(len(pictures)))
        generated_pictures += 1
        pic = load_img(resized_pictures_path + '/' + picture)
        pic_array = img_to_array(pic)
        pic_array = pic_array.reshape((1,) + pic_array.shape)
        count = 0
        for _ in generator.flow(pic_array, batch_size=BATCH_SIZE,
                                save_to_dir=train_dir_path,
                                save_prefix=PHOTO_PREFIX, save_format=OUTPUT_FORMAT):
            count += 1
            if count == multiplier:
                break


def get_model():
    result = Sequential()
    result.add(
        Conv2D(96, kernel_size=11, strides=1, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_COLORS)))
    result.add(BatchNormalization())
    result.add(MaxPooling2D((2, 2)))
    result.add(Conv2D(128, (9, 9), strides=1, activation='relu', padding="same"))
    result.add(BatchNormalization())
    result.add(MaxPooling2D(2, strides=2))
    result.add(Conv2D(256, (7, 7), strides=1, activation='relu', padding="same"))
    result.add(BatchNormalization())
    result.add(MaxPooling2D((2, 2)))
    result.add(Conv2D(256, (5, 5), strides=1, activation='relu', padding="same"))
    result.add(BatchNormalization())
    result.add(Conv2D(512, (3, 3), strides=1, activation='relu', padding="same"))
    result.add(BatchNormalization())
    result.add(MaxPooling2D(2, strides=2))
    result.add(Flatten())
    result.add(Dense(2048, activation='relu'))
    result.add(Dropout(0.75))
    result.add(Dense(1000, activation='relu'))
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

    plt.figure(figsize=(10, 5))  # Adjust the figure size if needed

    plt.subplot(1, 2, 1)  # Create a subplot for the accuracy
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)  # Create a subplot for the loss
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'y', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()  # Adjust spacing between subplots

    plt.show()


def demo():
    model = models.load_model(TRAINED_MODEL)
    # predicting images
    path = preprocessing(TEST_IMAGE_PATH)
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
    model.save(TRAINED_MODEL, include_optimizer=True)
    plot_graph(hist)


def preprocessing(image_path):
    selected_image = Image.open(image_path)

    # Resize the image while maintaining the aspect ratio
    selected_image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)

    # Create a new image with white background
    new_image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))

    # Calculate the position to paste the resized image
    position = ((IMAGE_SIZE - selected_image.size[0]) // 2, (IMAGE_SIZE - selected_image.size[1]) // 2)

    # Paste the resized image onto the new image
    new_image.paste(selected_image, position)

    # Save the new image
    image_name = image_path.split('/')[-1]
    image_name = 'preprocessed_' + image_name
    output_path = os.path.join('/'.join(image_path.split('/')[:-1]), image_name)
    new_image.save(output_path)

    print(f"Resized and saved {image_name}")
    return output_path


if not os.path.exists(NO_HELMET_TRAINING):
    augment_pictures(MULTIPLIER, NO_HELMET_INPUT, NO_HELMET_TRAINING)
if not os.path.exists(HELMET_TRAINING):
    augment_pictures(MULTIPLIER, HELMET_INPUT, HELMET_TRAINING)
if not os.path.exists(TRAINED_MODEL):
    save_model()
demo()
