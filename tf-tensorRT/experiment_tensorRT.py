import keras
from benchmark_trt import benchmark_trt
from datasets.loaderCIFAR10 import load


# CIFAR10 models
densenet121 = "./models_tensorRT/CIFAR10/DenseNet121"
googlenet = "./models_tensorRT/CIFAR10/GoogLeNet"
mobilenet = "./models_tensorRT/CIFAR10/MobileNetV2"
resnet20 = "./models_tensorRT/CIFAR10/ResNet20"
resnet44 = "./models_tensorRT/CIFAR10/ResNet44"


train, test = load()
dataset = train

benchmark_trt(densenet121, dataset, batch=64)
benchmark_trt(googlenet, dataset, batch=64)

