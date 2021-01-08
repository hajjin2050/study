from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), strides= 1 ,padding='same' ,input_shape = (10,10,1)))
model.add(MaxPooling2D(pool_size= (2,3))) #=>default:2
model.add(Conv2D(9, (2,2),padding='valid')) 
# model.add(Conv2D(9, (2,3))) 
# model.add(Conv2D(8 , 2)) #=> (2,2)를 2로 인식함
model.add(Flatten())
model.add(Dense(1))

model.summary()

# Model: "sequential"
# _________________________________________________________________# 2 x 2(필터 크기) x 1(입력 채널(RGB)) x 10(출력채널) + 10(출력 채널 bias) = 50

# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 9, 9, 10)          50  =>(input_dim x kernel_szie + bias)x filter
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 8, 8, 9)           369
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 7, 7, 8)           296
# _________________________________________________________________
# flatten (Flatten)            (None, 392)               0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 393
# =================================================================
# Total params: 1,108
# Trainable params: 1,108
# Non-trainable params: 0
# _________________________________________________________________
# PS C:\study\study> 