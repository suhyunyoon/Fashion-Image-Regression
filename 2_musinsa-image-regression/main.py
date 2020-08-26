from utils import Model
import numpy as np

musinsa_regression = Model()

# musinsa_data폴더의 데이터로 학습 + 결과 출력
musinsa_regression.train()

# 사진 예측
print(musinsa_regression.predict(np.array([])))