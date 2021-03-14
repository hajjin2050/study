#실습
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train,_),(x_test,_) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_tset = x_test.reshape(10000, 784)/255

#맼썸노이즈~
x_train_noised = x_train + np.random.normal(0, 0.1, size= x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size= x_test.shape)

x_train_noised = np.clip(x_Train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_Train_noised, a_min=0, a_max=1)

print(x_train_noised.shape)
print(x_test_noised.shape)

x_train_noised_in = x_train_noised.reshape(60000, 28, 28, 1)
x_test_noised_in = x_test_noised.reshape(10000, 28, 28, 1)

#cae모델 만들어봅시다
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten, Conv2DTranspose, LeakyReLU, Dropout, UpSampling2D, BatchNormaliztion, MaxPooling2D

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, strides=1, padding = 'same', input_shape=(28,28,1)))
    model.add(MaxPooling2D(2))
    # model.add(BatchNormalization())
    model.add(Conv2D(16, 2, 1))
    model.add(UpSampling2D((2, 2))) # 맥스풀 했으니 업 해줘야 한다.
    model.add(Flatten())
    model.add(Dense(units=hidden_layer_size/2))
    model.add(Dense(units=hidden_layer_size))
    model.add(Dense(units=784, activation='sigmoid'))
    return model
# ------------------------------------------------------------------------------------------------

model = autoencoder(hidden_layer_size=154)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train_noised_in, x_train, epochs=10)

# ------------------------------------------------------------------------------------------------

# 그림으로 그려보자
output = model.predict(x_test_noised_in)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20,7))

# 이미지 다섯 개를 무작위로 고르자
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력 데이터) 이미지를 맨 위에 그리자
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('NOISE', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더로 복원한 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()