import keras
from benchmark import benchmark_trt, benchmark
from loaderCIFAR10 import load
from pruning_structural import structural_pruning
from convert_to_TensorRT import convert_to_TensorRT

train, test = load()
dataset = train


'''
# CIFAR10 models
densenet121 = "./models_tensorRT/CIFAR10/DenseNet121"
densenet121_BASE = "./models/CIFAR10/densenet/DenseNet121.keras"

#densenet_STRUCTURAL_2_4 = structural_pruning(densenet121_BASE, "./models_pd/CIFAR10", "structural_2_4", (2, 4))

structural = "./models_pd/CIFAR10/structural_2_4.keras"
model = keras.models.load_model(structural)
model.save("./models_pd/CIFAR10/structural_2_4")
convert_to_TensorRT(
    "./models_pd/CIFAR10/structural_2_4", 
    "./models_tensorRT"
    )

densenet121_tensorRT = "./models_tensorRT/CIFAR10/structural_2_4"

benchmark_trt(densenet121, dataset, batch=64)
benchmark_trt(densenet121_tensorRT, dataset, batch=64)
'''


BASE = "./models_pd/CIFAR10/DenseNet121"
PRUNED = "./models_pd/CIFAR10/structural_2_4"

convert_to_TensorRT(
    BASE, 
    "./models_tensorRT"
    )

convert_to_TensorRT(
    PRUNED, 
    "./models_tensorRT"
    )

BASE_trt = "./models_tensorRT/CIFAR10/DenseNet121"
PRUNED_trt = "./models_tensorRT/CIFAR10/structural_2_4"

benchmark_trt(BASE_trt, dataset, batch=64)
benchmark_trt(PRUNED_trt, dataset, batch=64)
