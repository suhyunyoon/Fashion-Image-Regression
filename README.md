# All-one-1
- 서버에서 사용되는 AI모델을 훈련하고 예측하는 코드
- 서버에서 모델의 기능들을 간단히 함수로 호출할 수 있는 인터페이스 코드
- 현재는 /2_musinsa-image-regression/ 의 코드를 메인 모델로 사용한다.

## 1. fashion-MNIST random value regression
무신사 데이터를 크롤링하기 전 이미지 회귀 프로젝트를 전반적으로 이해하기 위한 실습(지금은 사용하지 않음)

- fashion-MNIST의 데이터에 정규분포를 따르는 랜덤한 값[0, 1]을 labeling

- keras을 사용해 VGG-Net11, 13을 참고한 모델의 끝에 하나의 값을 출력하는 유닛을 배치에 regression 문제를 해결하고자 함

- 이미지와 랜덤한 값의 관계를 찾지 못하고 데이터의 평균으로 수렴하는 결과가 나왔다.


## 2. MUSINSA Image Regression

상품 이미지가 input데이터로 주어질 때, 해당 상품의 조회수를 예측하는 모델

무신사에서 크롤링한 약 28,000개의 데이터로 학습시켰으며, keras를 사용해 CNN 모델을 구현했다.

### Contents
--------------
#### main.py
딥러닝 서버(AWS G3)에서 모델의 기능을 함수로 호출하는 코드.

#### utils.py
모델의 기능을 함수단위로 구현.
- 데이터 읽기
- 데이터 전처리
- 모델 학습(hyperparameter 조정)
- 학습 결과 plotting
- 추가적인 데이터로 전이학습
- 이미지 따른 조회수 예측

등의 함수들이 정의되어 있다.

#### models/CNN.py
VGG16 구조를 참고한 CNN 구현 코드.
이미지를 가공한 4차원 벡터(N x 60 x 60 x 3)를 input으로, 조회수 값 하나를 output으로 가진다.

#### musinsa_data/
크롤링한 상품 데이터들을 local 환경에서 저장한다.
"단화.csv", "셔츠.csv" 등의 table data와
"단화/1.jpg", "단화/1368.jpg" 등의 image data로 저장된다.

## history
[Train Records Document(/history/2/README.md)](https://git.swmgit.org/swmaestro/all-one-1/blob/master/history/2/README.md)
- 모델을 훈련하며 훈련 결과를 기록해 놓는 directory
- 구조를 계속해서 바꿔나가며 개선중이다.

/history/2/README.md 스크린샷
<div>
 <img src="/history/2/images/screenshot.png"/>
</div>