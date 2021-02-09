import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D

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
)
test_datagen = ImageDataGenerator(rescale=1./255)
#테스트 할 때는 많은 데이터가 필요없으니 전처리만!

#flow 또는 flow_from_directory 구성
#flow_from_directory를 통과하면 x,y가 생성됨.
#train_generater
xy_train = train_datagen.flow_from_directory(
    "C:/data/image/brain/train",
    target_size =(150, 150) ,
    batch_size = 200,
    class_mode = 'binary'
    ,save_to_dir = "C:/data/image/brain_generator/train"#'binary'에서는 앞에가 0이되면 뒤에가 알아서 1이됨(0~1 로 수렴)
)               #(80,150,150,1)
#증폭은 안되고 데이터 통으로 변환.
#Found 160 images belonging to 2 classes.

xy_test= test_datagen.flow_from_directory(
    "C:/data/image/brain/test",
    target_size =(150, 150) ,
    batch_size = 5,
    class_mode = 'binary'
    ,save_to_dir = "C:/data/image/brain_generator/test"#'binary'에서는 앞에가 0이되면 뒤에가 알아서 1이됨(0~1 로 수렴)
)
#Found 160 images belonging to 2 classes.

print(xy_train[0][0])
print(xy_train[0][1])
print(xy_train[0][1].shape)