import keras


# CIFAR10
densenet121 = "./models/CIFAR10/densenet/DenseNet121.keras"
model = keras.models.load_model(densenet121)
model.save("./models_pd/CIFAR10/DenseNet121")

googlenet = "./models/CIFAR10/googlenet/GoogLeNet.keras"
model = keras.models.load_model(googlenet)
model.save("./models_pd/CIFAR10/GoogLeNet")

mobilenet = "./models/CIFAR10/mobilenet/MobileNetV2.keras"
model = keras.models.load_model(mobilenet)
model.save("./models_pd/CIFAR10/MobileNetV2")

resnet20 = "./models/CIFAR10/resnet/ResNet20.keras"
model = keras.models.load_model(resnet20)
model.save("./models_pd/CIFAR10/ResNet20")

resnet44 = "./models/CIFAR10/resnet/ResNet44.keras"
model = keras.models.load_model(resnet44)
model.save("./models_pd/CIFAR10/ResNet44")


# CIFAR100
densenet121 = "./models/CIFAR100/densenet/DenseNet121.keras"
model = keras.models.load_model(densenet121)
model.save("./models_pd/CIFAR100/DenseNet121")

googlenet = "./models/CIFAR100/googlenet/GoogLeNet.keras"
model = keras.models.load_model(googlenet)
model.save("./models_pd/CIFAR100/GoogLeNet")

resnet18 = "./models/CIFAR100/resnet/ResNet18.keras"
model = keras.models.load_model(resnet18)
model.save("./models_pd/CIFAR100/ResNet18")

# GTSRB
densenet121 = "./models/GTSRB/densenet/DenseNet121.keras"
model = keras.models.load_model(densenet121)
model.save("./models_pd/GTSRB/DenseNet121")

