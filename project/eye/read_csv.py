import numpy as np
import csv

 #데이터 셋 만들기
def read_csv(path):
    width = 34
    height = 26
    dims = 1
    # 34 x 26 / 회색이미지

  with open(path,'r') as f:
    #read the scv file with the dictionary format
    reader = csv.DictReader(f)
    rows = list(reader)

  # 이미지와 
  imgs = np.empty((len(list(rows)),height,width, dims),dtype=np.uint8)
  tgs = np.empty((len(list(rows)),1))

  for row,i in zip(rows,range(len(rows))):
    #리스트 형식을 이미지형식으로 바꿔주기
    img = row['image']
    img = img.strip('[').strip(']').split(', ') # strip 함수는 제일 앞 또는 끝의 내용이 인자와 일치 할 경우 삭제해줌.
    im = np.array(img,dtype=np.uint8)
    im = im.reshape((height, width))
    im = np.expand_dims(im, axis=2) #expand_dims 는 차원을 확장시켜주는 함수 
    imgs[i] = im

    #the tag for open is 1 and for close is 0
    tag = row['state']
    if tag == 'open':
      tgs[i] = 1
    else:
      tgs[i] = 0

  #데이터셋을 섞어줌 
  index = np.random.permutation(imgs.shape[0]) #permutation 을 이용해서 순열을 구함
  imgs = imgs[index]
  tgs = tgs[index]

  return imgs, tgs


