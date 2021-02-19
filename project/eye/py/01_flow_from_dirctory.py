# ImageDataGeneator의  FIT_GENERATOR 사용해서완성
import PIL.Image as pilimg
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.tools import module_util as _module_util
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

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    "C:\data\p_project\eye\dataset\closed", # 눈감은 데이터셋 불러오기
    target_size =(26, 34) ,
    batch_size = 8,
    class_mode = 'binary'#'binary'에서는 앞에가 0이되면 뒤에가 알아서 1이됨(0~1 로 수렴)
    , subset = 'training'
)

xy_val= train_datagen.flow_from_directory(
    "C:\data\p_project\eye\dataset\open", #눈뜬 데이터 셋 불러오기
    target_size =(26, 34) ,
    batch_size = 8,
    class_mode = 'binary'#'binary'에서는 앞에가 0이되면 뒤에가 알아서 1이됨(0~1 로 수렴)
    , subset = 'validation'
)

# print(xy_train[0][0].shape)
# print(xy_train[0][1].shape)

model = Sequential()
model.add(Conv2D(32 ,(3,3), input_shape = (26,34,3)))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

#검사할 데이터를 넣어보자.
filepath='C:/data/p_project/eye/dataset/test_img/1.jpg'
image = pilimg.open(filepath)
image_data = image.resize((26,34))
image_data = np.array(image_data)
image_data = image_data.reshape(1,26,34,3)
answer = [0]
no_answer = [1]


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

print("결과")
y_predict = model.predict(image_data)