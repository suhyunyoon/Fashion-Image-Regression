# 학습 결과 기록

- Musinsa-pageView-regression 모델 튜닝을 위한 학습 기록
- Model architecture별로 정리
- Convolution 구조는 VGG16 구조를 참고
	- Convolution stride: 3x3
	- Maxpool: 2x2
	- FC layer: 4096->1
- 사진 순서는 Loss / Train Data / Test Data 순.
- 22번 사진부터는 전처리 시 제일 작은 데이터를 제외, 기존 log-scale후 arctan 추가, 크롤링 된 이산 데이터들을 랜덤으로 연속값으로 변환

## Conv2 + FC1
|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|96,321||

### param 수가 적은데도 overfitting 발생, 단일값으로 회귀하지는 않음
<div>
	<img width="33%" src="/before/history/2/images/loss%20(10).png"/>
	<img width="33%" src="/before/history/2/images/train%20(10).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(10).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|200|relu, linear|mae|96,321||

<div>
	<img width="33%" src="/before/history/2/images/loss%20(31).png"/>
	<img width="33%" src="/before/history/2/images/train%20(31).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(31).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|740|relu, linear|mae|96,321| 마지막 layer dropout 0.5, early stop|

<div>
	<img width="33%" src="/before/history/2/images/loss%20(32).png"/>
	<img width="33%" src="/before/history/2/images/train%20(32).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(32).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|1000|relu, linear|mae|96,321| dropout 0.3, He-normalization |
||||| 첫 pooling: Average pooling |

<div>
	<img width="33%" src="/before/history/2/images/loss%20(38).png"/>
	<img width="33%" src="/before/history/2/images/train%20(38).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(38).png"/>
</div>


## Conv2 + FC2
|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|235,976,513||

<div>
	<img width="33%" src="/before/history/2/images/loss%20(8).png"/>
	<img width="33%" src="/before/history/2/images/train%20(8).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(8).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|200|relu, linear|mae|235,976,513||

<div>
	<img width="33%" src="/before/history/2/images/loss%20(15).png"/>
	<img width="33%" src="/before/history/2/images/train%20(15).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(15).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|200|relu, linear|mse|235,976,513||

<div>
	<img width="33%" src="/before/history/2/images/loss%20(16).png"/>
	<img width="33%" src="/before/history/2/images/train%20(16).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(16).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|57,640,721|FC layer(1000->1)
|||||마지막 layer dropout 0.5|

<div>
	<img width="33%" src="/before/history/2/images/loss%20(11).png"/>
	<img width="33%" src="/before/history/2/images/train%20(11).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(11).png"/>
</div>

## Conv2 + FC4
|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|256,851,729||

<div>
	<img width="33%" src="/before/history/2/images/loss%20(7).png"/>
	<img width="33%" src="/before/history/2/images/train%20(7).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(7).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|256,851,729| 전처리 개선, 마지막 dropout 0.5 |

<div>
	<img width="33%" src="/before/history/2/images/loss%20(22).png"/>
	<img width="33%" src="/before/history/2/images/train%20(22).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(22).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|200|relu, linear|mae|256,851,729| 전처리 개선, 마지막 dropout 0.5 |

<div>
	<img width="33%" src="/before/history/2/images/loss%20(23).png"/>
	<img width="33%" src="/before/history/2/images/train%20(23).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(23).png"/>
</div>

## Conv2 + FC5
|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|273,633,041||

<div>
	<img width="33%" src="/before/history/2/images/loss%20(6).png"/>
	<img width="33%" src="/before/history/2/images/train%20(6).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(6).png"/>
</div>

## Conv4 + FC3
|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|122,327,057|마지막 layer dropout 0.5

<div>
	<img width="33%" src="/before/history/2/images/loss%20(12).png"/>
	<img width="33%" src="/before/history/2/images/train%20(12).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(12).png"/>
</div>

## Conv4 + FC4
|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|139,108,369||

<div>
	<img width="33%" src="/before/history/2/images/loss%20(1).png"/>
	<img width="33%" src="/before/history/2/images/train%20(1).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(1).png"/>
</div>

## Conv4 + FC5
|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|362,800,977| FC(10000->4096->1) |

<div>
	<img width="33%" src="/before/history/2/images/loss%20(29).png"/>
	<img width="33%" src="/before/history/2/images/train%20(29).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(29).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|200|relu, linear|mae|362,800,977| FC(10000->4096->1) |
|||||마지막 layer부터 dropout 0.5 + 0.25 * 3|

<div>
	<img width="33%" src="/before/history/2/images/loss%20(28).png"/>
	<img width="33%" src="/before/history/2/images/train%20(28).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(28).png"/>
</div>

## Conv7 + FC2
|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|14,281,489|FC layer(1000->1)

<div>
	<img width="33%" src="/before/history/2/images/loss%20(9).png"/>
	<img width="33%" src="/before/history/2/images/train%20(9).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(9).png"/>
</div>

## Conv7 + FC4
|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|73,999,121||

<div>
	<img width="33%" src="/before/history/2/images/loss%20(2).png"/>
	<img width="33%" src="/before/history/2/images/train%20(2).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(2).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|73,999,121|마지막 layer dropout 0.25 + l2 reg 0.001

### 단일값 회귀.
<div>
	<img width="33%" src="/before/history/2/images/loss%20(3).png"/>
	<img width="33%" src="/before/history/2/images/train%20(3).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(3).png"/>
</div>

## Conv7 + FC5
|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|90,780,433||

<div>
	<img width="33%" src="/before/history/2/images/loss%20(4).png"/>
	<img width="33%" src="/before/history/2/images/train%20(4).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(4).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|1000|relu, linear|mae|90,780,433||

<div>
	<img width="33%" src="/before/history/2/images/loss%20(5).png"/>
	<img width="33%" src="/before/history/2/images/train%20(5).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(5).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|90,780,433| 전처리 개선 |
|||||마지막 layer부터 dropout 0.5 + 0.25 * 3|

<div>
	<img width="33%" src="/before/history/2/images/loss%20(26).png"/>
	<img width="33%" src="/before/history/2/images/train%20(26).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(26).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mse|90,780,433||

<div>
	<img width="33%" src="/before/history/2/images/loss%20(13).png"/>
	<img width="33%" src="/before/history/2/images/train%20(13).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(13).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mse|90,780,433| 전처리 개선 |
|||||마지막 layer부터 dropout 0.5 + 0.25 * 3|

<div>
	<img width="33%" src="/before/history/2/images/loss%20(25).png"/>
	<img width="33%" src="/before/history/2/images/train%20(25).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(25).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|200|relu, linear|mse|90,780,433| 전처리 개선|

<div>
	<img width="33%" src="/before/history/2/images/loss%20(24).png"/>
	<img width="33%" src="/before/history/2/images/train%20(24).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(24).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|200|relu, linear|mse|90,780,433||

<div>
	<img width="33%" src="/before/history/2/images/loss%20(13).png"/>
	<img width="33%" src="/before/history/2/images/train%20(13).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(13).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|200|relu, linear|mse|90,780,433| 전처리 개선 |
|||||마지막 layer부터 dropout 0.5 + 0.25 * 3|

<div>
	<img width="33%" src="/before/history/2/images/loss%20(27).png"/>
	<img width="33%" src="/before/history/2/images/train%20(27).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(27).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mse|90,780,433| fc모두 dropout 0.3, He-normalization |
||||| 첫 pooling: Average pooling, conv layer batch norm |

<div>
	<img width="33%" src="/before/history/2/images/loss%20(37).png"/>
	<img width="33%" src="/before/history/2/images/train%20(37).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(37).png"/>
</div>

## Conv10 + FC4
|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|60,080,449||

### Conv13+FC5보다 param수가 적은데 단일값으로 회귀하지 않음
<div>
	<img width="33%" src="/before/history/2/images/loss%20(18).png"/>
	<img width="33%" src="/before/history/2/images/train%20(18).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(18).png"/>
</div>

## Conv10 + FC5
|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|64,174,353||

### Conv13+FC5보다 param수가 적은데 단일값으로 회귀하지 않음
<div>
	<img width="33%" src="/before/history/2/images/loss%20(19).png"/>
	<img width="33%" src="/before/history/2/images/train%20(19).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(19).png"/>
</div>

## Conv13 + FC3
|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|20,913,937||

### 단일값 회귀.
<div>
	<img width="33%" src="/before/history/2/images/loss%20(17).png"/>
	<img width="33%" src="/before/history/2/images/train%20(17).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(17).png"/>
</div>

## Conv13 + FC4
|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|37,695,249||

### 단일값 회귀.
<div>
	<img width="33%" src="/before/history/2/images/loss%20(30).png"/>
	<img width="33%" src="/before/history/2/images/train%20(30).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(30).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|37,695,249| 마지막부터 dropout 0.5 + 0.25 * 3|

### 단일값 회귀.
<div>
	<img width="33%" src="/before/history/2/images/loss%20(33).png"/>
	<img width="33%" src="/before/history/2/images/train%20(33).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(33).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|37,695,249| 마지막 dropout 0.5 |
||||| He-normalization |

<div>
	<img width="33%" src="/before/history/2/images/loss%20(34).png"/>
	<img width="33%" src="/before/history/2/images/train%20(34).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(34).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|37,695,249| 마지막부터 dropout 0.5 + 0.25 * 3 |
||||| He-normalization |

<div>
	<img width="33%" src="/before/history/2/images/loss%20(35).png"/>
	<img width="33%" src="/before/history/2/images/train%20(35).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(35).png"/>
</div>

## Conv13 + FC5
|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|67,163,969||

### 단일값 회귀.
<div>
	<img width="33%" src="/before/history/2/images/loss%20(20).png"/>
	<img width="33%" src="/before/history/2/images/train%20(20).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(20).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|1000|relu, linear|mae|67,163,969||

### 단일값 회귀.
<div>
	<img width="33%" src="/before/history/2/images/loss%20(21).png"/>
	<img width="33%" src="/before/history/2/images/train%20(21).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(21).png"/>
</div>

|epoch|activation|loss|params|etc|
|--|--|--|--|--|
|50|relu, linear|mae|37,695,249| FC(10000->4096->1), 마지막부터 dropout 0.5 + 0.25 * 3 |
||||| He-normalization |

<div>
	<img width="33%" src="/before/history/2/images/loss%20(36).png"/>
	<img width="33%" src="/before/history/2/images/train%20(36).png"/>
	<img width="33%" src="/before/history/2/images/predict%20(36).png"/>
</div>
