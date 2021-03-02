from tensorflow.keras.applications import VGG16

model = VGG16(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3))

model.trainable = False
model.summary()
print(len(model.weights))
print(len(model.trainable_weights))