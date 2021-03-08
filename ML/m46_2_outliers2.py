# 실습
# outliers1 을 행렬형태로 적요ㅗㅇ할 수 있도록 수정

import numpy as np
aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],[10000,20000,3,40000,50000,60000,70000,8,90000,100000]])
aaa = aaa.transpose()
print(aaa.shape) #(10, 2)

def outliers(data_out):
    allout = []
    for i in range(data_out.shape[1]):
        quartile_1, q2, quartile_3 = np.percentile(data_out,[25, 50, 75])
        print("1시분위 : ", quartile_1)
        print("q2 :", q2)
        print("3사분위 :",quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1- (iqr *1.5)
        upper_bound = quartile_1- (iqr *1.5)
        print('lower_bound:', lower_bound)
        print('upper_bound:', upper_bound)
        a = np.where((data_out[:,i]>upper_bound)|(data_out[:,i]<lower_bound))
        allout.append(a)
    return np.array(allout)

outlier_loc = outliers(aaa)

print("이상치의 위치:",outlier_loc)


import matplotlib.pyplot as plt
plt.boxplot(aaa[:,1])
plt.show()
