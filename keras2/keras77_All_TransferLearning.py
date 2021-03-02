from tensorflow.keras.applications import VGG16,VGG19, Xception
from tensorflow.keras.applications import ResNet101,ResNet101V2, ResNet152,ResNet152V2,ResNet50,ResNet50V2
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3,InceptionResNetV2
from tensorflow.keras.applications import MobileNet,MobileNetV2
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6,EfficientNetB7
from tensorflow.keras.applications import NASNetMobile,DenseNet121

model = VGG19()
model = Xception()
model = ResNet50()
model = ResNet101()
model = InceptionV3()
model = InceptionResNetV2()
model = DenseNet121()
model = NASNetMobile()
model = MobileNetV2()
model = EfficientNetB0()




model.trainable =False
model.summary()