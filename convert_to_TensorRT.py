import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt


def model_name(path : str) -> str:
    arg = path.split("/")[-1].split(".")[0]
    arg = arg.split(".")[0]
    return arg

def dataset_name(path : str) -> str:
    arg = path.split("/")[-2]
    return arg


def convert_to_TensorRT(pd_model_path : str, output_saved_model_dir : str) -> str:

    SAVE_PATH = output_saved_model_dir + "/" + dataset_name(pd_model_path) + "/" + model_name(pd_model_path)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=pd_model_path,
        precision_mode=trt.TrtPrecisionMode.FP16
    )
    converter.convert()
    converter.save(SAVE_PATH)
    

if __name__ == "__main__":

    # CIFAR10 models
    densenet121 = "./models_pd/CIFAR10/DenseNet121"
    googlenet = "./models_pd/CIFAR10/GoogLeNet"
    mobilenet = "./models_pd/CIFAR10/MobileNetV2"
    resnet20 = "./models_pd/CIFAR10/ResNet20"
    resnet44 = "./models_pd/CIFAR10/ResNet44"

    convert_to_TensorRT(densenet121, "./models_tensorRT")
    #convert_to_TensorRT(googlenet, "./models_tensorRT")
    #convert_to_TensorRT(mobilenet, "./models_tensorRT")
    #convert_to_TensorRT(resnet20, "./models_tensorRT")
    #convert_to_TensorRT(resnet44, "./models_tensorRT")
    

