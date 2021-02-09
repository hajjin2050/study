# ImageDataGeneator의  FIT_GENERATOR 사용해서완성

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten

train_datagen = ImageDataGenerator(
    rescale=1./255, # 1/255로 스케일링하여 0-1 범위로 변환시켜 학습시키기에 알맞은 모델로 변환.
    horizontal_flip = True, #
    vertical_flip = True,
    width_shift_range =0.1,
    height_shift_range = 0.1,
    rotation_range = 5, # 이미지 회전범위
    zoom_range = 1. ,
    shear_range= 0.7,
    fill_mode = 'nearest' #이미지 증폭할때 근처 비슷한애들로 채워주겠다( padding 같은개념)
    ,validation_split = 0.25
)
'''
- rotation_range: 이미지 회전 범위 (degrees)
- width_shift, height_shift: 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 
                                (원본 가로, 세로 길이에 대한 비율 값)
- rescale: 원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 
            모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 
            그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 
            이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.
- shear_range: 임의 전단 변환 (shearing transformation) 범위
- zoom_range: 임의 확대/축소 범위
- horizontal_flip`: True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다. 
    원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.
- fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
'''

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

model = Sequential()
model.add(Conv2D(32 ,(3,3), input_shape = (150,150,3)))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.5, verbose = 1)
filepath = 'c:/data/modelcheckpoint/keras62_1_checkpoint_{val_loss:.4f}-{epoch:02d}.hdf5'
cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')

history = model.fit_generator(xy_train, steps_per_epoch=93, epochs=500, validation_data=xy_val, validation_steps=31,
callbacks=[es])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print('acc : ', acc[-1])
print('val_acc : ', val_acc[:-1])