from keras.models import Sequential, load_model
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Activation

# VGG16
def get_model():
    self.model = Sequential()
    self.model.add(Conv2D(filters=64,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding='same',
                          activation='relu',
                          kernel_initializer='he_normal',
                          input_shape=(60, 60, 3)))
    self.model.add(Conv2D(filters=64,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          activation='relu',
                          kernel_initializer='he_normal',
                          padding='same'))
    self.model.add(AveragePooling2D(pool_size=(2, 2),
                                    strides=(2, 2)))

    self.model.add(Conv2D(filters=128,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation='relu',
                    kernel_initializer='he_normal',
                    padding='same'))
    self.model.add(Conv2D(filters=128,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation='relu',
                    kernel_initializer='he_normal',
                    padding='same'))
    self.model.add(MaxPooling2D(pool_size=(2, 2),
                          strides=(2, 2)))
    #self.model.add(Dropout(0.25))

    self.model.add(Conv2D(filters=256,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation='relu',
                    kernel_initializer='he_normal',
                    padding='same'))
    self.model.add(Conv2D(filters=256,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation='relu',
                    kernel_initializer='he_normal',
                    padding='same'))
    self.model.add(Conv2D(filters=256,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation='relu',
                    kernel_initializer='he_normal',
                    padding='same'))
    self.model.add(MaxPooling2D(pool_size=(2, 2),
                          strides=(2, 2)))
    #self.model.add(Dropout(0.5))

    self.model.add(Conv2D(filters=512,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation='relu',
                    kernel_initializer='he_normal',
                    padding='same'))
    self.model.add(Conv2D(filters=512,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation='relu',
                    kernel_initializer='he_normal',
                    padding='same'))
    self.model.add(Conv2D(filters=512,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation='relu',
                    kernel_initializer='he_normal',
                    padding='same'))
    self.model.add(MaxPooling2D(pool_size=(2, 2),
                          strides=(2, 2)))

    self.model.add(Conv2D(filters=512,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation='relu',
                    kernel_initializer='he_normal',
                    padding='same'))
    self.model.add(Conv2D(filters=512,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation='relu',
                    kernel_initializer='he_normal',
                    padding='same'))
    self.model.add(Conv2D(filters=512,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation='relu',
                    kernel_initializer='he_normal',
                    padding='same'))
    self.model.add(MaxPooling2D(pool_size=(2, 2),
                          strides=(2, 2)))

    self.model.add(Flatten())
    # self.model.add(Dense(units=4096, activation='relu'))
    self.model.add(Dense(units=10000, activation='relu', kernel_initializer='he_normal'))
    self.model.add(Dropout(0.3))
    self.model.add(Dense(units=4096, activation='relu', kernel_initializer='he_normal'))
    self.model.add(Dropout(0.3))
    self.model.add(Dense(units=4096, activation='relu', kernel_initializer='he_normal'))
    self.model.add(Dropout(0.3))
    self.model.add(Dense(units=1000, activation='relu', kernel_initializer='he_normal'))
    self.model.add(Dropout(0.3))
    self.model.add(Dense(units=1, activation='linear',
                         kernel_initializer='he_normal'))

    self.model.compile(loss='mae',
                       optimizer='adam',
                       metrics=['mae', 'mse'])
    return model