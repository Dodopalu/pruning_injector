import keras

resnet20 = "./models_pd/CIFAR10/ResNet20"

model = keras.models.load_model(resnet20)
model.summary()