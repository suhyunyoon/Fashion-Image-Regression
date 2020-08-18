# 학습 결과 기록

- Musinsa-pageView-regression 모델 튜닝을 위한 학습 기록
- Model architecture별로 정리
- Convolution 구조는 VGG Net구조를 참고
	- Convolution stride: 3x3
	- Maxpool: 2x2
	- FC layer: 4096->1

## Conv2 + FC1
### 96,321 params
|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mae||
![loss](10) ![test data](10) ![train data](10)
### param 수가 적은데도 overfitting 발생, 단일값으로 회귀하지는 않음

## Conv2 + FC2
### 235,976,513 params
### 57,640,721 params(1000->1)
|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mae||
![loss](8) ![test data](8) ![train data](8)

|epoch|activation|loss|etc|
|--|--|--|--|
|200|relu, linear|mae||
![loss](15) ![test data](15) ![train data](15)

|epoch|activation|loss|etc|
|--|--|--|--|
|200|relu, linear|mse||
![loss](16) ![test data](16) ![train data](16)

|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mae|FC layer(1000->1)
||||마지막 layer dropout 0.5|
![loss](11) ![test data](11) ![train data](11)

## Conv2 + FC4
### 256,851,729 params
|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mae||
![loss](7) ![test data](7) ![train data](7)

## Conv2 + FC5
### 273,633,041 params
|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mae||
![loss](6) ![test data](6) ![train data](6)

## Conv4 + FC3
### 122,327,057 params
|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mae|마지막 layer dropout 0.5
![loss](12) ![test data](12) ![train data](12)

## Conv4 + FC4
### 139,108,369 params
|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mae||
![loss](1) ![test data](1) ![train data](1)

## Conv7 + FC2
### 14,281,489 params(1000->1)
|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mae|FC layer(1000->1)
![loss](9) ![test data](9) ![train data](9)

## Conv7 + FC4
### 73,999,121 params
|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mae||
![loss](2) ![test data](2) ![train data](2)

|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mae|마지막 layer dropout 0.25 + l2 reg 0.001
![loss](3) ![test data](3) ![train data](3)
### 단일값 회귀.

## Conv7 + FC5
### 90,780,433 params
|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mae||
![loss](4) ![test data](4) ![train data](4)

|epoch|activation|loss|etc|
|--|--|--|--|
|1000|relu, linear|mae||
![loss](5) ![test data](5) ![train data](5)

|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mse||
![loss](13) ![test data](13) ![train data](13)

|epoch|activation|loss|etc|
|--|--|--|--|
|200|relu, linear|mse||
![loss](14) ![test data](14) ![train data](14)

## Conv10 + FC4
### 60,080,449 params
|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mae||
![loss](18) ![test data](18) ![train data](18)
### 20번보다 param수가 적은데 단일값으로 회귀하지 않음

## Conv10 + FC5
### 64,174,353 params
|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mae||
![loss](19) ![test data](19) ![train data](19)
### 20번보다 param수가 적은데 단일값으로 회귀하지 않음

## Conv13 + FC3
### 20,913,937 params
|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mae||
![loss](17) ![test data](17) ![train data](17)
### 단일값 회귀.

## Conv13 + FC5
### 67,163,969 params
|epoch|activation|loss|etc|
|--|--|--|--|
|50|relu, linear|mae||
![loss](20) ![test data](20) ![train data](20)
### 단일값 회귀.
