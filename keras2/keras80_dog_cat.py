#이미지는
#data/image/vgg 에 고양이 개 라이언 슈트 넣을것
#파일명 : dog1.jpg cat1,jpg lion,jpg suit1.jpg

from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array # 이미지를 넘파이 배열로 바꿔준다!
import numpy as np

img_dog = load_img('C:/data/image/vgg/dog1.jpg', target_size = (224,224)) # 사이즈도 지정가능
img_cat = load_img('C:/data/image/vgg/cat1.jpg', target_size = (224,224))
img_ryan = load_img('C:/data/image/vgg/lion1.jpg', target_size = (224,224))
img_suit = load_img('C:/data/image/vgg/suit1.jpg', target_size = (224,224))

plt.imshow(img_cat) # 이미지 잘 불러왔나 확인!
plt.show()

print(img_suit) # 로드 이미지 했을땐 케라스로 임포트 해서 케라스 형식이다! 
# <PIL.Image.Image image mode=RGB size=224x224 at 0x1CD822F85B0>
# 그래서 image_to_array 를 사용해 배열로 바꿔준다

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_ryan = img_to_array(img_ryan)
arr_suit = img_to_array(img_suit)

# print(arr_dog) # [254. 254. 255.]]]
# print(type(arr_suit)) <class 'numpy.ndarray'>
# print(arr_dog.shape) (224, 224, 3)

# 이렇게 이미지를 불러오면 RGB 형태이다
# 근데 VGG16 에 넣을때는 BGR 형태로 넣어야한다
# RGB -> BGR
from tensorflow.keras.applications.vgg16 import preprocess_input
# 알아서 vgg16 에 맞춰 전처리를 해준다
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_ryan = preprocess_input(arr_ryan)
arr_suit = preprocess_input(arr_suit)

# print(arr_dog)
# print(arr_dog.shape) (224, 224, 3)
# 쉐이프는 동일, R 과 G 의 위치만 바뀌었다

arr_input = np.stack([arr_dog, arr_cat, arr_ryan, arr_suit]) # np.stack 은 배열을 합쳐주는 역할
# print(arr_input.shape) (4, 224, 224, 3) ## 순서대로 dog, cat, ryan, suit 가 합쳐진 배열


#2. 모델구성   ## 훈련 시킬게 아니라 모델을 그대로 쓸것이다
model = VGG16()
results = model.predict(arr_input)

# print(results)
# print('results.shape : ', results.shape)
'''
# [[1.4992932e-09 4.9452953e-10 5.1459503e-11 ... 4.8396742e-10     
#   1.6936048e-07 1.1220930e-06]
#  [4.3583753e-07 1.2146568e-06 4.4076779e-07 ... 5.3686909e-07
#   3.3494583e-05 3.7730621e-05]
#  [9.2846369e-07 4.9175487e-06 1.0787576e-06 ... 8.0677171e-07
#   3.9812530e-05 3.4896409e-04]
#  [3.1493435e-08 9.4961639e-10 2.1516704e-09 ... 1.1728181e-10
#   2.4805676e-08 2.1393834e-07]]
# results.shape :  (4, 1000)    << 1000 은 imagenet 에서 분류할수 있는 카테고리 수
'''

# 이걸 확인 어떻게 할까? 확인하는것 또한 제공해주겠지
# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions # 해독한다! > 예측한걸 해석한다

decode_results = decode_predictions(results)
print('===========================================')
print('results[0] : ', decode_results[0])
print('===========================================')
print('results[1] : ', decode_results[1])
print('===========================================')
print('results[2] : ', decode_results[2])
print('===========================================')
print('results[3] : ', decode_results[3])
print('===========================================')
