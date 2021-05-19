# fashion-image-regression-with-multi-feature-extraction
## SW마에스트로 2020 

서버 내 조회수 예측 모델부분 코드

수집된 이미지와 조회수 등의 상품정보를 전처리,

Weakly Supervised Segmentation 모델을 학습 또는 Mask-RCNN모델을 사용해 mask를 만들고,

Mask와 Raw 이미지 데이터로 상품, 배경의 색상, 모델의 노출도 등의 피처를 추출하고,

추출된 피처로 조회수를 회귀하는 모델을 학습하는 부분으로 분리


## Codes
### `load_and_save_image.py`
~~~
python load_and_save_image.py
~~~
수집된 .jpg 이미지 데이터를 raw 이미지 데이터로 변환하 .hdf5 파일로 저장

items.csv에서 상품정보를 읽어 사진이 없는 상품은 제외하고 index를 저장

### `create_seg_mask_rcnn.py`
~~~
git clone https://github.com/matterport/Mask_RCNN.git
cd /Mask_RCNN/
python setup.py install
cd ../
python create_seg_mask_rcnn.py
~~~
Segmentation, Detection 모델인 [Mask-RCNN](https://arxiv.org/abs/1703.06870)을 사용하여

상품 이미지에서 상품의 영역만 나타내는 mask를 추출 후 masks.hdf5파일로 저장

### `extract_feature.py`
~~~
python extract_feature.py
~~~
이미지 raw 데이터와 추출된 mask 데이터를 통해 상품의 색상, 배경의 색상, 복잡도, 이미지내 상품이 차지하는 사이즈, 모델의 노출도를 추출하여 feature.csv파일로 저장

### `extract_feature_cupy.py`
~~~
python extract_feature_cupy.py
~~~
연산속도를 높이기 위해 numpy 연산을 gpu로 할당하는 cupy를 사용한 버전.

### `CNN_regression.py`
~~~
python CNN_regression.py
~~~
raw image를 input으로 조회수를 예측하는 CNN모델. ResNet50 모델을 변형하여 사용

### `regression_model.py`
~~~
python regression_model.py
~~~
추출된 feature(feature.csv)를 통해 조회수를 예측하는 머신러닝 모델

## Directories
### `opencv/`
상품 이미지의 모델 정보를 추출하기 위한 haar-cascade 기법에 사용되는 xml 파일이 저장된다.

### `temp/`
local 환경에 저장되는 학습에 필요한 크롤링된 임시 데이터, 학습 후 local에서 저장할 모델, 결과 값들

*items.csv*와 같이 학습에 필요한 상품 정보 데이터,

*images.hdf5*, *masks.hdf5*와 같이 학습에 필요한 이미지 numpy 배열 데이터,

*features.csv*와 같은 상품 이미지에서 추출된 hand-crafted feature 데이터,

*model.h5*와 같은 저장된 학습된 모델 파라미터 등이 저장

### `before/`
([README](/before/README.md))

중간평가 이전의 모델, 코드들. 다양한 시도를

**1_fashion-MNIST-regression/**

fashion-MNIST dataset으로 회귀값을 예측할 수 있는지에 대한 코드

**2_musinsa-image-regression/**

크롤링한 약 28,000개의 무신사 데이터셋으로 상품 조회수값을 예측할 수 있는지에 대한 코드
변형된 VGG16, 19등의 모델 사용

**history/**

*README.md*에 나와있는 parameter tuning의 시도들 저장
