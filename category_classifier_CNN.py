from PIL import Image
import os, glob
import numpy as np
import pandas as pd
from tensorflow.keras.utils import HDF5Matrix
import h5py

import matplotlib.pyplot as plt
import matplotlib

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

def create_CNN():
    base = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    fc = Flatten()(base.output)
    #fc = Dense(units=1000, activation='relu', kernel_initializer='he_normal')(fc)
    fc = Dense(units=1, activation='linear', kernel_initializer='he_normal')(fc)

    model = Model(inputs=base.inputs, outputs=fc)

    model.compile(loss='sparse_catetorial_crossentropy',
                       optimizer='adam',
                       metrics=['mae', 'mse'])
    model.summary()
    return model

class CNN_classifier:
    def __init__(self, item_path='temp/', image_file='images.hdf5', item_file='items.csv'):
        self.item_path = item_path
        self.image_file = image_file
        self.item_file = item_file

        self.model = Sequential(ResNet50(include_top=True, weights=None, input_shape=(224, 224, 3), classes=2))




def train(train_images, train_labels, valid_images, valid_labels):

    # PARAMETERS
    epoch = 5
    batch_size = 32
    early = EarlyStopping(monitor='val_mae', min_delta=0, patience=10, verbose=1, mode='auto')

    # BUILD YOUR MODEL
    model = create_CNN()

    model.summary()

    hist = model.fit(train_images, train_labels, epochs=epoch, batch_size=batch_size, shuffle='batch',
                     validation_data=(valid_images, valid_labels), verbose=1)

    plt.plot(hist.history['loss'], 'b-', label="training")
    plt.plot(hist.history['val_loss'], 'r:', label="validation")
    plt.legend()
    plt.savefig(os.path.join(item_path, 'temp/loss.png'), dpi=300)
    plt.show()

    # Make prediction data frame
    valid_pred = model.predict(valid_images)
    pred_labels = valid_labels
    # index = np.argsort(pred_labels)
    index = np.arange(valid_pred.shape[0])
    scatter = plt.scatter(index, pred_labels, c='b', s=0.1, label='test label')
    scatter = plt.scatter(index, valid_pred, c='r', s=0.1, label='predict')
    plt.legend()
    plt.savefig(os.path.join(item_path, 'temp/predict.png'), dpi=300)
    plt.show()

    h5_file = os.path.join(item_path, 'temp/model.h5')
    model.save(h5_file)

    return model

if __name__ == '__main__':
    # 파일 경로 지정
    current_dir = os.path.join(os.getcwd(), './')
    item_path = os.path.join(current_dir, 'temp/')
    item_csv = {}

    item_file = os.path.join(item_path, 'items.csv')
    item_csv = pd.read_csv(item_file, encoding='utf-8').fillna('0')
    # korean to index
    #item_csv['category'] = item_list[i]
    image_file = os.path.join('images.hdf5')

    '''
    # shuffle
    index = np.argsort(Y)[:-1]
    np.random.shuffle(index)
    #rand_index = np.random.permutation(num_X)
    X = X[index] / 255.0
    # 좀더 선형으로 만들기
    Y = np.log(Y[index] + 1)
    Y = (Y - Y.min()) / (Y.max() - Y.min())
    #Y = np.arctan(Y - Y.mean())
    '''
    Y = item_csv['pageView'].astype(int)

    with h5py.File(item_path + image_file, 'a') as f:
        num_X = len(f['image'])
        # random index
        #index = np.random.permutation(num_X)
        Y = item_csv[item_csv['id'].isin(f['index'])]['pageView'].astype(int).to_numpy()
        # 선형으로
        Y = np.log(Y + 1)
        Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

    # train
    train_images = HDF5Matrix(item_path+image_file, 'image', end=int(num_X * 0.9))
    train_labels = Y[:int(num_X * 0.9)]
    valid_images = HDF5Matrix(item_path+image_file, 'image', start=int(num_X * 0.9))
    valid_labels = Y[int(num_X * 0.9):]

    a = train(train_images, train_labels, valid_images, valid_labels)
