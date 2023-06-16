import tensorflow as tf
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
        for batch in train_data_generator.flow(pic_array, batch_size=1, save_to_dir=path + '/generated_images',
                                               save_prefix='generated', save_format='jpg'):
            count += 1
            if count == multiplier:
                break


def get_model():
    model = tf.keras.models.Sequential([
        # 1st conv
        tf.keras.layers.Conv2D(64, kernel_size=11, strides=8, activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, strides=2),
        # 2nd conv
        tf.keras.layers.Conv2D(256, (11, 11), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        # 3rd conv
        tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        # 4th conv
        tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        # 5th Conv
        tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
        # To Flatten layer
        tf.keras.layers.Flatten(),
        # To FC layer 1
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # To FC layer 2
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # get description
    model.summary()
    return model


def compile_model(model):
    from keras.optimizers.legacy import Adam
    model.compile(
        optimizer=Adam(learning_rate=0.005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


def fit_model(model, train_generator, validation_generator):
    hist = model.fit(train_generator,
                     validation_split=0.2,
                     steps_per_epoch=train_generator.samples // batch_size,
                     validation_data=validation_generator,
                     validation_steps=validation_generator.samples // batch_size,
                     epochs=50)


augment_pictures(4, 'data_copy/nohelmet_b')
model = get_model()
compile_model(model)
datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    directory='/Users/danielkosc/Documents/MUNI/Spring2023/ML/project/data_copy/Final_data',
    target_size=(128, 128),
    batch_size=20,
    class_mode='binary',
    shuffle=True,
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    directory='/Users/danielkosc/Documents/MUNI/Spring2023/ML/project/data_copy/Final_data',
    target_size=(128, 128),
    batch_size=20,
    class_mode='binary',
    shuffle=True,
    subset='validation'
)
fit_model(model, train_generator, validation_generator)
