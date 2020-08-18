from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# VGG16
def get_model():
    model = Sequential()
    model.add(Conv2D(filters=64,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding='same',
                          activation='relu',
                          input_shape=(60, 60, 3)))
    model.add(Conv2D(filters=64,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          activation='relu',
                          padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2)))

    model.add(Conv2D(filters=128,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          activation='relu',
                          padding='same'))
    model.add(Conv2D(filters=128,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          activation='tanh',
                          padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Conv2D(filters=256,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          activation='relu',
                          padding='same'))
    model.add(Conv2D(filters=256,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          activation='relu',
                          padding='same'))
    model.add(Conv2D(filters=256,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          activation='relu',
                          padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(filters=512,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          activation='relu',
                          padding='same'))
    model.add(Conv2D(filters=512,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          activation='relu',
                          padding='same'))
    model.add(Conv2D(filters=512,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          activation='relu',
                          padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2)))
    '''
    model.add(Conv2D(filters=512,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation='relu',
                    padding='same'))
    model.add(Conv2D(filters=512,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation='relu',
                    padding='same'))
    model.add(Conv2D(filters=512,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation='relu',
                    padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                          strides=(2, 2)))
    '''
    model.add(Flatten())
    # model.add(Dense(units=4096, activation='relu'))
    # model.add(Dense(units=4096, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dense(units=4096, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=1000, activation='relu'))
    model.add(Dense(units=1, activation='linear'))

    model.compile(loss='mae',
                       optimizer='adam',
                       metrics=['mae', 'mse'])
    return model