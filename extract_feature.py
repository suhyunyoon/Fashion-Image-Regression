import numpy as np
import cv2
import h5py
import pandas as pd
from scipy.stats import entropy


class FeatureExtractor:
    def __init__(self, batch_size=10000, img_size=(224,224)):
        (self.image_w, self.image_h) = img_size
        self.batch_size = batch_size

        # opencv body detection model
        self.opencv_path = './opencv/'
        self.full_body_detector = cv2.CascadeClassifier(self.opencv_path + 'haarcascade_fullbody.xml')
        self.upper_body_detector = cv2.CascadeClassifier(self.opencv_path + 'haarcascade_upperbody.xml')
        self.lower_body_detector = cv2.CascadeClassifier(self.opencv_path + 'haarcascade_lowerbody.xml')
        self.face_detector = cv2.CascadeClassifier(self.opencv_path + 'haarcascade_frontalface_alt2.xml')

        # size(big, medium, small)
        self.size_degree = (0.55, 0.35)
        # classify color(HSV not RGB)
        self.colors = ((0, 0, 255, "white"),
                      (0, 0, 0, "black"),
                      (0, 0, 120, "grey"),
                      (176, 215, 190, "red"),
                      (23, 151, 220, "yellow"),
                      (35, 132, 139, "khaki"),
                      (105, 85, 180, "blue"),
                      (120, 64, 80, "navy"),
                      (13, 89, 100, "brown"),
                      (15, 113, 180, "beige"))

    # Main
    def extract_feature(self, image_file='temp/images.hdf5', masks_file='temp/masks.hdf5'):
        # check detector loaded
        if self.full_body_detector.empty() or self.upper_body_detector.empty() or self.lower_body_detector.empty() or self.face_detector.empty():
            print('Classifier XML not Loaded!')
            return None

        # load data
        try :
            with h5py.File(image_file, 'r') as f, h5py.File(masks_file, 'r') as m:
                num_X = len(f['image'])

                ##########################################################
                #num_X = 1000
                #self.batch_size = 1000
                self.a = np.zeros((num_X, 3), dtype=np.float32)

                # initialization
                color = np.empty(num_X, dtype=np.dtype("U10"))
                bg_color = np.empty(num_X, dtype=np.dtype("U10"))
                comp = np.zeros(num_X, dtype=np.uint8)
                size = np.zeros(num_X, dtype=np.uint8)
                model_type = np.zeros(num_X, dtype=np.uint8)

                for i in range(0, num_X, self.batch_size):
                    # lazy load data(hdf5)
                    X = f['image'][i:i+self.batch_size]
                    masks = m['mask'][i:i+self.batch_size]
                    print('{}~{} Data Loaded'.format(i, i+len(masks)))

                    # 상품, 배경 색감 추출
                    color[i:i+self.batch_size], bg_color[i:i+self.batch_size] = self.extract_color(X, masks)
                    # 배경의 복잡도
                    comp[i:i+self.batch_size] = self.image_complexity(X, masks)
                    # 상품 크기 비율
                    size[i:i+self.batch_size] = self.estimate_size(masks)
                    # 모델의 노출도
                    model_type[i:i+self.batch_size] = self.detect_body(X)

                df = pd.DataFrame({'id': f['index'], 'color': color, 'bg_color': bg_color, 'bg_complexity': comp, 'size': size, 'model_type': model_type})
                print('Feature Extracted.')
        except :
            df = None
        return df

    def nearest_color(self, x):
        return min(self.colors, key=lambda color: sum((c - x_) ** 2 for c, x_ in zip(color, x)))[-1]

    def extract_color(self, X, masks):
        # reshape
        masks_ = np.reshape(masks, masks.shape[:-1]).astype(np.bool)
        # extract color
        num_X = len(X)
        color = np.zeros(num_X, dtype=np.dtype("U10"))
        bg_color = np.zeros(num_X, dtype=np.dtype("U10"))

        # convert BGR(opencv default) to HSV
        X_ = np.reshape(X, (num_X * self.image_w, self.image_h, 3))
        X_ = cv2.cvtColor(X_, cv2.COLOR_BGR2HSV)
        X_ = np.reshape(X_, X.shape)

        # test
        for i, (x, m) in enumerate(zip(X_, masks_)):
            #print(self.nearest_color(colors, np.mean(X[i][masks[i]], axis=0).astype(int)))
            color[i] = self.nearest_color(np.mean(x[m], axis=0).astype(np.uint8))
            #print(self.nearest_color(colors, np.mean(X[i][np.logical_not(masks[i])], axis=0).astype(int)))
            bg_color[i] = self.nearest_color(np.mean(x[np.logical_not(m)], axis=0).astype(np.uint8))

        print('extract_color Done.')
        return color, bg_color

    # 배경 복잡도 (1: 복잡함 0:안복잡함)
    def image_complexity(self, X, masks):
        # 복잡도 기준
        threshold = 25.0
        # reshape
        # masks = np.reshape(masks, masks.shape[:-1])

        R, G, B = X[:, :, :, 0:1], X[:, :, :, 1:2], X[:, :, :, 2:]

        # compute rg = R - G
        rg = np.absolute(R - G)

        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)

        _masks = np.logical_not(masks)
        # compute the mean and standard deviation of both `rg` and `yb` ()
        rb_masked = np.ma.masked_where(_masks, rg)
        (rb_mean, rb_std) = (np.mean(rb_masked, axis=(1, 2, 3)), np.std(rb_masked, axis=(1, 2, 3)))

        yb_masked = np.ma.masked_where(_masks, yb)
        (yb_mean, yb_std) = (np.mean(yb_masked, axis=(1, 2, 3)), np.std(yb_masked, axis=(1, 2, 3)))

        # combine the mean and standard deviations
        std_root = np.sqrt((rb_std ** 2) + (yb_std ** 2))
        mean_root = np.sqrt((rb_mean ** 2) + (yb_mean ** 2))

        # derive the "colorfulness" metric and return it
        complexity = std_root + (0.3 * mean_root)

        # 수치 수정
        print('image_complexity Done.')
        return (complexity >= threshold).astype(np.uint8)

    def estimate_size(self, masks):
        d = np.sqrt(np.sum(masks, axis=(1, 2, 3), dtype=np.uint32) / (masks.shape[1] * masks.shape[2]))
        # big: 2, medium: 1, small: 0
        size = np.zeros_like(d)
        size = np.where(d >= self.size_degree[1], 1, size)
        size = np.where(d >= self.size_degree[0], 2, size)

        print('estimate_size Done.')
        return size

    # 5: 전신 4: 얼굴빼고 전신 3: 상반신 2: 얼굴빼고 상반신 1: 하반신 0: 모델 없음
    def detect_body(self, X):
        # init
        num_X = len(X)
        model_type = np.zeros(num_X, dtype=np.uint8)

        # detect each image
        for i, x in enumerate(X):
            img = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            # full_body_detected
            full_body_detected = len(self.full_body_detector.detectMultiScale(img, 1.01, 3)) > 0
            #upper_body_detected
            upper_body_detected = len(self.upper_body_detector.detectMultiScale(img, 1.01, 3)) > 0
            #lower_body_detected
            lower_body_detected = len(self.lower_body_detector.detectMultiScale(img, 1.01, 3)) > 0

            if full_body_detected and (upper_body_detected or lower_body_detected):
                # 5: 전신(with face)
                if len(self.face_detector.detectMultiScale(img, 1.01, 3)) > 0:
                    model_type[i] = 5
                # 4: 얼굴빼고 전신
                else:
                    model_type[i] = 4
            elif upper_body_detected:
                # 3: 상반신(with face)
                if len(self.face_detector.detectMultiScale(img, 1.01, 3)) > 0:
                    model_type[i] = 3
                # 2: 얼굴빼고 상반신
                else:
                    model_type[i] = 2
            # 1: 하반신
            elif lower_body_detected:
                model_type[i] = 1
            # 0: 모델 없음
            else :
                model_type[i] = 0

        print('detect_body Done.')
        return model_type

if __name__=='__main__':
    item_dir = 'temp/'
    extractor = FeatureExtractor()
    # extract feature
    df = extractor.extract_feature(image_file=item_dir+'images.hdf5', masks_file=item_dir+'masks.hdf5')
    # save csv
    df.to_csv(item_dir + 'features.csv', sep=',', na_rep='NaN', encoding='utf-8')