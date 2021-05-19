(실패)1_fashion-MNIST-random-value-regression

1. fashion-MNIST의 데이터에 정규분포를 따르는 랜덤한 값[0, 1]을 labeling

2. keras을 사용해 VGG-Net11, 13을 참고한 모델의 끝에 하나의 값을 출력하는 유닛을 배치에 regression 문제를 해결하고자 함

3. 이미지와 랜덤한 값의 관계를 찾지 못하고 데이터의 평균으로 수렴하는 결과가 나왔다.