x_train = 0.5
y_train = 0.8

##############바꿔################
weight = 0.1
lr = 0.01
epoch = 500
#############바꿔바꿔####################

for itertion in range(epoch):
    y_predict = x_train * weight
    error = (y_predict - y_train)**2

    print("Error: " +str(error) +"\ty_predcit:"+ str(y_predict))

    up_y_predict = x_train * (weight + lr)
    up_error = (y_train - up_y_predict) ** 2

    down_y_predict = x_train * (weight - lr)
    down_error = (y_train - down_y_predict) ** 2 

    if(down_error<= up_error):
        weight = weight - lr
    if(down_error > up_error):
        weight = weight + lr

#loss = error = cost
