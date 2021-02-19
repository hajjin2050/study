# (3633, 34, 26, 3) (3633,)
# (1213, 34, 26, 3) (1213,)

# 실패 y shpae 의 컬럼을 0(감은상태), 1(뜬상태)로 만들어주려고함
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization,MaxPooling2D, Flatten
from sklearn.metrics import accuracy_score
import PIL.Image as pilimg
from PIL import Image

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
    "C:\data\p_project\eye\dataset",
    target_size = (34, 26),
    batch_size =100000,
    class_mode = 'binary',
    subset = 'training'
)
xy_test = train_datagen.flow_from_directory(
    "C:\data\p_project\eye\dataset",
    target_size = (34, 26),
    batch_size =100000,
    class_mode = 'binary',
    subset = 'validation'
)


np.save('C:/data/p_project/eye/npy/train_x.npy', arr=xy_train[0][0])
np.save('C:/data/p_project/eye/npy/train_y.npy', arr=xy_train[0][1])
np.save('C:/data/p_project/eye/npy/val_x.npy', arr=xy_test[0][0])
np.save('C:/data/p_project/eye/npy/val_y.npy', arr=xy_test[0][1])

x_train = np.load('C:/data/p_project/eye/npy/train_x.npy')
x_val = np.load('C:/data/p_project/eye/npy/val_x.npy')
y_train = np.load('C:/data/p_project/eye/npy/train_y.npy')
y_val = np.load('C:/data/p_project/eye/npy/val_y.npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
