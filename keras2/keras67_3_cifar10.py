# cifar10   flow를 구성해서 완성
# ImageDataGenerator    /   fit_generator

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.datasets import cifar10

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip = True,
    # vertical_flip = True,
    width_shift_range =0.1,
    height_shift_range = 0.1,
    # rotation_range = 5,
    # zoom_range = 1.2,
    shear_range= 0.7,
    fill_mode = 'nearest' #이미지 증폭할때 근처 비슷한애들로 채워주겠다( padding 같은개념)
    ,validation_split = 0.25
)
test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    "C:/data/image/gender",
    target_size =(150, 150) ,
    batch_size = 8,
    class_mode = 'binary'#'binary'에서는 앞에가 0이되면 뒤에가 알아서 1이됨(0~1 로 수렴)
    , subset = 'training'
)

xy_val= train_datagen.flow_from_directory(
    "C:/data/image/gender",
    target_size =(150, 150) ,
    batch_size = 8,
    class_mode = 'binary'#'binary'에서는 앞에가 0이되면 뒤에가 알아서 1이됨(0~1 로 수렴)
    , subset = 'validation'
)

print(xy_train[0][0].shape)
print(xy_train[0][1].shape)

x_train = np.load('../data/npy/cifar10_x_train.npy')
y_train = np.load('../data/npy/cifar10_y_train.npy')
x_test = np.load('../data/npy/cifar10_x_test.npy')
y_test = np.load('../data/npy/cifar10_y_test.npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
