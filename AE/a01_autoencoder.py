import numpy as np
from tensorflow.keras.datasets import mnist

# 오토인코더 - 비지도 학습, 차원축소에도 사용
# y값이 없다!! => x-> ㅁ ->x 의 구조
# 784, 엠니스트 데이터가 64 덴스레이어로 들어가고 다시 784, 로 나온다면
# 데이터 축소, 데이터 확장이 같이 이뤄진다

(x_train, _), (x_test, _) = mnist.load_data() # _ 은 쓰지 않을 변수, y 값을 사용하지 않을것이다!

#전처리
x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784)/255.

# print(x_train[0]) # 잘출력되는지 확인
# print(x_test[0])

#모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape = (784,))
encoded = Dense(64, activation = 'relu')(input_img)
decoded = Dense(784, activation = 'sigmoid')(encoded) # 데이터를 0~1로 수렴하게 전처리해줬으니 난 정말 시그모이드

autoencoder = Model(input_img, decoded)

autoencoder.summary()

# autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc']) # 왜 바이너리...?
# acc 넣고 실행해보자 > 엉망이다, loss 를 기준으로 잡자
autoencoder.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc']) # mse/ binary 둘 다 상관없음 (0~1로 수렴시켜줬으니까)

autoencoder.fit(x_train, x_train, epochs = 30, batch_size = 256, validation_split=0.2) # y 값 대신 x 값을 넣는다!!

decoded_imgs = autoencoder.predict(x_test)

# 데이터 시각화 해당 코드 실행시
# 1 2 3 4 5 원본
# 1 2 3 4 5 변형
# 이렇게 볼 수 있다!
import matplotlib.pyplot as plt 
n = 10
plt.figure(figsize = (20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False) # 이미지 옆의 눈금선을 FALSE
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()