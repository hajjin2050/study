import cv2
import numpy as np
from matplotlib import pyplot as plt

######################################################################################
# 이미지를 넘파이로 바꿔주는 과정!

aaa = []
for i in range(50000):
    image_path = f'../dacon/clean/clean_mnist/{str(i).zfill(5)}.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image2 = np.where((image <= 254) & (image != 0), 0, image)
    image2 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
    # image2 = cv2.medianBlur(src=image2, ksize= 5)
    image2 = np.asarray(image2).reshape(34,26,1)
    aaa.append(image2)

aaa = np.array(aaa)/255.

np.save('../dacon/npy/x_train.npy', arr = aaa)
print(aaa.shape)

######################################################################################
# 테스트도 동일하게!

aaa = []
for i in range(50000,55000):
    image_path = f'../dacon/clean/test_clean_mnist/{str(i).zfill(5)}.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image2 = np.where((image <= 254) & (image != 0), 0, image)
    image2 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
    image2 = cv2.medianBlur(src=image2, ksize= 5)
    image2 = np.asarray(image2).reshape(256,256,1)
    aaa.append(image2)

aaa = np.array(aaa)/255.

np.save('../dacon/npy/x_test.npy', arr = aaa)
print(aaa.shape)