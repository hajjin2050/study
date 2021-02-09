import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization,MaxPooling2D, Flatten

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip = True,
    # vertical_flip = True,
    width_shift_range =0.1,
    height_shift_range = 0.1,
    fill_mode = 'nearest',
    validation_split = 0.2505
)
test_datagen = ImageDataGenerator(rescale= 1./255)

xy_train = train_datagen.flow_from_directory(
    "C:/data/image/gender",
    target_size = (150, 150),
    batch_size =14,
    class_mode = 'binary',
    subset = 'training'
)
xy_val = train_datagen.flow_from_directory(
    "C:/data/image/gender",
    target_size = (150, 150),
    batch_size =14,
    class_mode = 'binary',
    subset = 'validation'
)

print(xy_train[0][0].shape)
print(xy_train[0][1].shape)

np.save('C:/data/image/gender/npy/keras66_train_x.npy', arr=xy_train[0][0])
np.save('C:/data/image/gender/npy/keras66_train_y.npy', arr=xy_train[0][1])
np.save('C:/data/image/gender/npy/keras66_val_x.npy', arr=xy_val[0][0])
np.save('C:/data/image/gender/npy/keras66_val_y.npy', arr=xy_val[0][1])

x_train = np.load('C:/data/image/gender/npy/keras66_train_x.npy')
x_val = np.load('C:/data/image/gender/npy/keras66_val_x.npy')

y_train = np.load('C:/data/image/gender/npy/keras66_train_y.npy')
y_val = np.load('C:/data/image/gender/npy/keras66_val_y.npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

model = Sequential()
model.add(Conv2D(128, 3, padding='same', activation='relu', input_shape=(150,150,3)))
model.add(BatchNormalization())
model.add(Conv2D(64,3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(3))
model.add(Conv2D(32,3, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 30, factor = 0.5, verbose = 1)
filepath = 'c:/data/modelcheckpoint/keras62_1_checkpoint_{val_loss:.4f}-{epoch:02d}.hdf5'
cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')
history = model.fit(x_train,y_train, epochs=500, validation_data=(x_val,y_val),callbacks=[es])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print("acc:", acc[0])

import matplotlib.pyplot as plt
epochs = len(acc)
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, acc, label='train')
ax.plot(x_axis, val_acc, label='val')
ax.legend()
plt.ylabel('acc')
plt.title('acc')
# plt.show()


fig, ax = plt.subplots()
ax.plot(x_axis, loss, label='train')
ax.plot(x_axis, val_loss, label='val')
ax.legend()
plt.ylabel('loss')
plt.title('loss')
plt.show()