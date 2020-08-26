from PIL import Image
import os, glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

from keras.models import load_model
from keras.callbacks import EarlyStopping

from models import CNN

class Model:
    model = None
    def __init__(self, p = os.getcwd()):
        self.item_list = {
            '단화': 0,
            '데님': 1,
            '미니 원피스': 2,
            '미니스커트': 3,
            '미디 원피스': 4,
            '반팔': 5,
            '셔츠': 6,
            '슬랙스': 7,
            '카디건': 8,
            '카라 티셔츠': 9,
            '후드 집업': 10
        }
        # 파일 경로 지정
        #self.current_dir = os.path.join(os.getcwd(), 'drive/My Drive/Colab Notebooks')
        self.current_dir = p
        self.item_path = os.path.join(self.current_dir, 'musinsa_data')
        # 한글 데이터를 숫자로(추후 크롤링 코드 수정해서 숫자 값으로 받아오게)
        # 100만 회 이상 -> 1000000
        def numtext_to_num(p):
            mult = 1
            p = p.strip().replace(' ', '')
            p = p.replace('회이상', '')
            if p.endswith('천'):
                p = p.replace('천', '')
                mult = 1000
            elif p.endswith('만'):
                p = p.replace('만', '')
                mult = 10000
            ret = int(float(p) * mult)
            return ret
        # numpy vectorize 함수화
        self._t2n = np.vectorize(numtext_to_num)

    def _read_csv(self):
        item_csv = {}
        for i in self.item_list:
            item = os.path.join(self.item_path, i + '.csv')
            try:
                item_csv[i] = pd.read_csv(item, encoding='CP949').fillna('0')
                print('Reading ' + item + '...')
                # korean to index
                item_csv[i]['category'] = self.item_list[i]
            except:
                print('Cannot read ' + item)
        return item_csv

    def _get_label(self, name='pageView'):
        Y = []
        for i in self.item_list:
            Y += list(self.item_csv[i]['pageView'])
        Y = self._t2n(Y)
        return Y

    def _read_image(self):
        # 이미지 크기
        image_w = 60
        image_h = 60
        # 모든 이미지를 X에 저장
        # image, ndarray
        X = np.empty((0, image_w, image_h, 3))
        # 카테고리 별로 이미지 파일 읽음
        for item in self.item_list:
            # load saved .npy file
            try:
                # 이미지 numpy 배열이 저장되어 있는지 확인
                tempX = np.load(os.path.join(self.item_path, item + '.npy'))
                print('Reading ' + item + '.npy...')
            # read all .jpg file
            except:
                tempX = []
                image_path = os.path.join(self.item_path, item, '*.jpg')
                files = glob.glob(image_path)
                # 모든 이미지 파일 읽음
                print('Reading jpg files...')
                for i, f in enumerate(files):
                    img = Image.open(f)
                    img = img.convert("RGB")
                    img = img.resize((image_w, image_h))
                    # numpy
                    data = np.asarray(img)
                    tempX.append(data)
                    if i % 100 == 0:
                        print(i)
                tempX = np.array(tempX)
                print('save ' + item + '.npy...')
                np.save(os.path.join(self.item_path, item + '.npy'), tempX)
            # concat
            X = np.concatenate((X, tempX), axis=0)
        return X

    def _split_data(self, X, Y):
        num_X = len(X)
        train_ratio = 0.8
        train_images = X[:int(num_X * train_ratio)]
        train_labels = Y[:int(num_X * train_ratio)]
        test_images = X[int(num_X * train_ratio):]
        test_labels = Y[int(num_X * train_ratio):]
        return train_images, train_labels, test_images, test_labels

    def _preprocessing(self, X, Y):
        # shuffle
        index = np.argsort(Y)[:-1]
        np.random.shuffle(index)
        # rand_index = np.random.permutation(num_X)
        X = X[index] / 255.0
        # 좀더 선형으로 만들기
        Y = np.log(Y[index] + 1)
        Y = (Y - Y.min()) / (Y.max() - Y.min())
        # Y = np.arctan(Y - Y.mean())
        return self._split_data(X, Y)

    def _load_model(self):
        h5_file = os.path.join(self.current_dir, 'model.h5')
        if self.model is None:
            try:
                self.model = load_model(h5_file)
            except:
                self.model = CNN.get_model()

    def _plot(self, hist, test_images, test_labels):
        # 의미없는 첫 epoch 제외(loss가 너무 크게 나옴)
        plt.plot(hist.history['loss'][1:], 'b-', label="training")
        plt.plot(hist.history['val_loss'][1:], 'r:', label="validation")
        plt.legend()
        plt.savefig(os.path.join(self.current_dir, 'loss.png'), dpi=300)
        plt.show()

        # Make prediction data frame
        test_pred = self.model.predict(test_images)
        pred_labels = test_labels
        index = np.argsort(pred_labels)
        # index = np.arange(test_pred.shape[0])
        scatter = plt.scatter(np.arange(test_pred.shape[0]), pred_labels[index], c='b', s=0.1, label='test label')
        scatter = plt.scatter(np.arange(test_pred.shape[0]), test_pred[index], c='r', s=0.1, label='predict')
        plt.legend()
        plt.savefig(os.path.join(self.current_dir, 'predict.png'), dpi=300)
        plt.show()

    def _train(self, train_images, train_labels, test_images, test_labels):
        # PARAMETERS
        epoch = 200
        batch_size = 512
        early = EarlyStopping(monitor='loss', min_delta=0, patience=300, verbose=1, mode='auto')

        # BUILD MODEL
        self._load_model()

        #%%time
        hist = self.model.fit(train_images, train_labels, epochs=epoch, batch_size=batch_size,
                         validation_data=(test_images[:-2000], test_labels[:-2000]), verbose=1, callbacks=[early])
        # plot loss and test scatter
        self._plot(hist, test_images[-2000:], test_labels[-2000:])

        # save model
        h5_file = os.path.join(self.current_dir, 'model.h5')
        self.model.save(h5_file)

    def predict(self, test_images, test_labels):
        if self.model is None:
            print('Model is not trained. call train() first.')
            return np.array([])
        else :
            test_pred = self.model.predict(test_images)
            return test_pred

    def train(self):
        item = self._read_csv()
        ########################
