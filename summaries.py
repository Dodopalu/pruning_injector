import keras

resnet20 = "./models_pd/CIFAR10/ResNet20"
vgg = "./models/flatten_Vgg11_bn.keras"


model = keras.models.load_model(vgg)
model.summary()