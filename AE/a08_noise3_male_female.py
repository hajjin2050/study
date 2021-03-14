#keras67_1 남자 여자에 잡음넣어서 기미 주근꺠 여드름ㅇ르 제거하시오import numpy as np

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization,MaxPooling2D, Flatten
import random

x_train = np.load('C:/data/image/gender/x_train.npy')
x_test = np.load('C:/data/image/gender/x_test.npy')
x_train = x_train / 255.
x_test = x_test / 255.

#멬썸 노이즈~~
x_train_noised = x_train + np.random.normal(0, 0.1, size = x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size = x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min = 0, a_max= 1).reshape(x_train_noised.shape[0],256,256,3)
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max = 1).reshape(x_test_noised.shape[0],256,256,3)


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, BatchNormalization, Conv2DTranspose, LeakyReLU, Dropout

def autoencoder():
    model = Sequential()
    model.add(Conv2D(256, 3, activation= 'relu', padding= 'same', input_shape = (256,256,3)))
    model.add(Conv2D(256, 5, activation= 'relu', padding= 'same'))
    model.add(Conv2D(3, 3, padding = 'same', activation= 'sigmoid'))

    return model

# def autoencoder2():
#     inputs = Input(shape=(28,28,1))
#     x = Conv2D(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(inputs)
#     x = BatchNormalization()(x)
#     x = LeakyReLU()(x)
#     x_1 = x

#     x = Conv2D(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU()(x)
#     x_2 = x


#     x = Conv2DTranspose(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(x+x_2)
#     x = Dropout(0.4)(x)
#     x = LeakyReLU()(x)
#     x = x

#     x = Conv2DTranspose(filters=1,kernel_size=4,strides=2,use_bias=False,padding='same', activation='sigmoid')(x+x_1)
#     outputs = x
#     model = Model(inputs = inputs,outputs=outputs)


#     return model

model = autoencoder()

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train_noised, x_train, epochs = 10, batch_size = 8)

output = model.predict(x_test_noised)


from matplotlib import pyplot as plt
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
        plt.subplots(3, 5, figsize = (20, 7))

# 이미지 다섯개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다!!
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(256, 256, 3), cmap = 'gray')
    if i==0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(256, 256, 3), cmap = 'gray')
    if i==0:
        ax.set_ylabel('NOISE', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# # 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(256, 256, 3), cmap = 'gray')
    if i==0:
        ax.set_ylabel('OUTPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()