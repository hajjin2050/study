# ImageDataGeneator의  FIT_GENERATOR 사용해서완성

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip = True,
    vertical_flip = True,
    width_shift_range =0.1,
    height_shift_range = 0.1,
    rotation_range = 5,
    zoom_range = 1.2,
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

import matplotlib.pyplot as plt
epochs = len(acc)
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, acc, label='train')
ax.plot(x_axis, acc, label='val')
ax.legend()#주석
plt.ylabel('acc')
plt.title('acc')
plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, loss, label='train')
ax.plot(x_axis, val_loss, label='val')
ax.legend()#주석
plt.ylabel('loss')
plt.title('loss')
plt.show()

import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread(f'C:/data/image/gender/female ({i}).jpg')
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()