import keras

resnet20 = "./models_tensorRT/CIFAR10/ResNet20"

model = keras.models.load_model(resnet20)
model.summary()