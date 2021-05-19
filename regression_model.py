from extract_feature import FeatureExtractor
#from extract_feature_cupy import FeatureExtractorCuPy
from load_and_save_image import read_images
from create_seg import gen_mask

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation

class PageViewPredictModel:
    def __init__(self, img_size=(224,224)):
        (self.image_w, self.image_h) = img_size
        self.model = None

        self.extractor = FeatureExtractor(10000, (224,224))
        #self.extractor = FeatureExtractorCuPy(self.item_dir, self.images_hdf5, self.masks_hdf5, 10000, (224,224))

    # True(default): 학습, False: 테스트(inference)
    #def trainMode(self, f):
    #    self.train = f

    # log-scale preprocessing
    def transform_pageView(self, data=None, reverse=False):
        if reverse:
            data = np.exp(data).astype(np.uint64)
        else:
            data = np.log(data + 1)
            #return (data - data.min()) / (data.max() - data.min())
        return data

    def get_categorial_dummies(self, df):
        colorDtype = pd.api.types.CategoricalDtype(categories=["white", "black", "grey", "red", "yellow",
                                                               "khaki", "blue", "navy", "brown", "beige"])
        compDtype = pd.api.types.CategoricalDtype(categories=[0, 1])
        sizeDtype = pd.api.types.CategoricalDtype(categories=[0, 1, 2])
        modelDtype = pd.api.types.CategoricalDtype(categories=[0, 1, 2, 3, 4, 5])
        brandDtype = pd.api.types.CategoricalDtype(categories=[0, 1, 2])

        df['color'] = df['color'].astype(colorDtype)
        df['bg_color'] = df['bg_color'].astype(colorDtype)
        df['bg_complexity'] = df['bg_complexity'].astype(compDtype)
        df['size'] = df['size'].astype(sizeDtype)
        df['model_type'] = df['model_type'].astype(modelDtype)
        df['brand_size'] = df['brand_size'].astype(brandDtype)

        return pd.get_dummies(df)

    def getColumns(self, df=None, item_file='temp/items.csv', label=True, transfer=False):
        # get item info
        if label:
            #try:
            item = pd.read_csv(item_file, sep=',').fillna(0)
            # drop 0 pageView
            item = item.drop(item.loc[item['pageView'] == 0].index)

            if not transfer:
                # drop invalid category
                item = item.drop(item.loc[item['category'].str.find('>') == -1].index)
                # 브랜드별 상품당 조회수 평균
                brand_mean = item.groupby('brand').mean()['pageView']
                # 0: 초소형 브랜드, 1: 중형 브랜드, 2: 대형 브랜드
                item['brand_size'] = 0
                item['brand_size'].where((brand_mean[item['brand']] < 80).values, 1, inplace=True)
                item['brand_size'].where((brand_mean[item['brand']] < 300).values, 2, inplace=True)
            else:
                # No Brand
                item['brand_size'] = 0

            # add(inner join)
            df = pd.merge(df, item[['id','brand_size','pageView']], left_on='id', right_on='id', how='inner')

            # label 전처리
            Y = df['pageView']
            Y = self.transform_pageView(Y)

            ################################
            #Y += np.random.randint(0,5,len(Y))
            ################################

            # remove id, pageView
            df = df.drop(['id','pageView'], axis=1)
            # one-hot encoding
            X = self.get_categorial_dummies(df)

           # No Brand
            if transfer:
                X['brand_size_1'] = 0
                X['brand_size_2'] = 0

            return X, Y

            #except Exception as err:
            #    print(err)
            #    return None
        # inference
        else:
            # No Brand
            df['brand_size'] = 0

            # remove id
            df = df.drop('id', axis=1)

            X = self.get_categorial_dummies(df)
            # 0: 초소형 브랜드, 1: 중형 브랜드, 2: 대형 브랜드
            X['brand_size_1'] = 0
            X['brand_size_2'] = 0
            return X

    def create_model(self, input_shape):
        model = Sequential()
        model.add(Dense(32, input_shape=(input_shape,), kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(8, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(1, activation='linear', kernel_initializer='he_normal'))

        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        return model

    def plot(self, pred, Y, s=0.1):
        plt.scatter(np.arange(len(pred)), Y[np.argsort(Y)], s=s)
        plt.scatter(np.arange(len(pred)), pred[np.argsort(Y)], s=s)
        plt.show()

    def train(self, epoch=100, transfer=False, replace=False, project_dir='temp/', image_dir='image/', image_file='temp/images.hdf5', mask_file='temp/masks.hdf5', item_file='temp/items.csv'):
        # read images
        if not os.path.exists(image_file):
            read_images(image_dir=project_dir+image_dir, file_name=image_file)

        # training seg Model
        if not transfer:
            ########################
            pass
            ########################
        # segmentation
        if not os.path.exists(mask_file):
            gen_mask(image_file=image_file, mask_file=mask_file)

        # get features
        #try:
        #    df = pd.read_csv(project_dir + 'features.csv', sep=',', index_col=0)
        #except:
            # extract feature from image, masks
        #    df = self.extractor.extract_feature()
        # extract feature from image, masks
        if (not transfer) and os.path.exists(project_dir + 'features.csv'):
            df = pd.read_csv(project_dir + 'features.csv', sep=',', index_col=0)
        else:
            df = self.extractor.extract_feature(image_file=image_file, masks_file=mask_file)
        # get training data
        X, Y = self.getColumns(df, item_file=item_file, transfer=transfer)

        # transfer learning
        if transfer:
            # load model
            if not self.model:
                self.model = self.create_model(len(X.keys()))
                try:
                    self.model.load_weights(project_dir + 'model_base.h5')
                except:
                    print('Pre-trained Model Not Exists!')

            # training
            hist = self.model.fit(X, Y, epochs=epoch, batch_size=1024, verbose=0)
            # save weight
            self.model.save_weights(project_dir + 'model_transfer_h5')

        # base learning
        else:
            self.model = self.create_model(len(X.keys()))
            # training
            hist = self.model.fit(X, Y, epochs=epoch, batch_size=1024, verbose=1)
            # save weight
            self.model.save_weights(project_dir + 'model_base.h5')

        # plotting
        self.plot(self.model.predict(X), Y, 1 if transfer else 0.1)
        print('Training Done.')

    def predict(self, project_dir='temp/', image_dir='image/'):
        image_file = project_dir + 'img_temp.hdf5'
        mask_file = project_dir + 'mask_temp.hdf5'
        # read images
        read_images(image_dir=project_dir+image_dir, file_name=image_file)
        # generate mask
        gen_mask(image_file=image_file, mask_file=mask_file)

        # extract feature from image, masks
        df = self.extractor.extract_feature(image_file=image_file, masks_file=mask_file)

        # get training data
        X = self.getColumns(df, label=False)

        if not self.model:
            self.model = self.create_model(len(X.keys()))
            self.model.load_weights(project_dir + 'model_transfer.h5')

        pred = self.model.predict(X)
        num_Y = len(pred)
        # 신뢰구간 내 오차 ( +- 1.96(95% z값) * std(Y) / sqrt(len(Y)))
        error = 1.96 * 1.15892 / num_Y
        pred_min, pred_max = pred - error, pred + error
        print('Predicted.')

        # return min max
        return self.transform_pageView(pred_min, reverse=True).reshape(len(pred)), self.transform_pageView(pred_max, reverse=True).reshape(len(pred))


if __name__ == '__main__':
    # Initialize Model
    model = PageViewPredictModel()

    # transfer: 사용자 데이터 받을 땐 True(전이학습)
    # project_dir: 프로젝트 base 디렉토리
    # image_dir: jpg파일들 저장된 디렉토리
    # image_file: 이미지 hdf5를 저장할 '경로+이름.hdf5'
    # mask_file: 마스크 이미지 hdf5를 저장할 '경로+이름.hdf5'
    # item_file: 사용자가 업로드한 csv 파일 '경로+이름.csv'

    #1 Base Train
    item_dir = 'temp/'
    image_dir = 'image/'
    model.train(epoch=20, transfer=False, project_dir=item_dir, image_dir = image_dir, image_file=item_dir+'images.hdf5', mask_file=item_dir+'masks.hdf5', item_file=item_dir+'items.csv')

    #2.1 Transfer Learning
    item_dir = 'temp/'
    image_dir = 'project01/image/'
    model.train(epoch=10, transfer=True, replace=True, project_dir=item_dir, image_dir = image_dir, image_file=item_dir+'project01/images.hdf5', mask_file=item_dir+'project01/masks.hdf5', item_file=item_dir+'items.csv')

    #2.2 Inference
    item_dir = 'temp/project01/test/'
    image_dir = 'image/'
    pred_min, pred_max = model.predict(project_dir=item_dir, image_dir = image_dir)

    print(pred_min, pred_max)