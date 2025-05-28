import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver

def model_name(path : str) -> str:
    arg = path.split("/")[-1].split(".")[0]
    arg = arg.split(".")[0]
    return arg

def dataset_name(path : str) -> str:
    arg = path.split("/")[-2]
    return arg


def convert_to_TensorRT(pd_model_path : str, output_saved_model_dir : str) -> str:

    SAVE_PATH = output_saved_model_dir + "/" + dataset_name(pd_model_path) + "/" + model_name(pd_model_path)

    # modify SPARSE_WEIGHTS flag

    
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=pd_model_path,
        precision_mode=trt.TrtPrecisionMode.FP16
    )



    converter.convert()
    converter.save(SAVE_PATH)


def convert_to_TensorRT_sparse_weights(pd_model_path : str, output_saved_model_dir : str) -> str:
    SAVE_PATH = output_saved_model_dir + "/" + dataset_name(pd_model_path) + "/" + model_name(pd_model_path)
    
    # Creare configurazione personalizzata per TensorRT
    conversion_params = trt.TrtConversionParams(
        precision_mode=trt.TrtPrecisionMode.FP16,
        # Utilizza un oggetto RewriterConfig personalizzato per configurare i flag TensorRT
        rewriter_config_template=None,  # Useremo custom_trt_rewriter_config
        use_dynamic_shape=False,
        max_workspace_size_bytes=1 << 30,  # 1GB
        minimum_segment_size=3,
        is_dynamic_op=True,
        maximum_cached_engines=1
    )
    
    # Configurazione personalizzata per TensorRT
    custom_trt_rewriter_config = config_pb2.RewriterConfig()
    custom_trt_rewriter_config.meta_optimizer_iterations = config_pb2.RewriterConfig.ONE
    optimizer = custom_trt_rewriter_config.custom_optimizers.add()
    optimizer.name = "TensorRTOptimizer"
    
    # Configurare i flag TensorRT incluso sparse_weights
    optimizer.parameter_map["precision_mode"].s = conversion_params.precision_mode.encode()
    optimizer.parameter_map["max_workspace_size_bytes"].i = conversion_params.max_workspace_size_bytes
    optimizer.parameter_map["minimum_segment_size"].i = conversion_params.minimum_segment_size
    optimizer.parameter_map["is_dynamic_op"].b = conversion_params.is_dynamic_op
    
    # Aggiungere il flag sparse_weights (imposta a true)
    optimizer.parameter_map["trt_engine_capability"].i = 0  # 0 = default
    optimizer.parameter_map["sparse_weights"].b = True
    
    # Passare la configurazione personalizzata
    conversion_params = trt.TrtConversionParams(
        precision_mode=trt.TrtPrecisionMode.FP16,
        rewriter_config_template=custom_trt_rewriter_config,
        use_dynamic_shape=False,
        max_workspace_size_bytes=1 << 30,
        minimum_segment_size=3,
        is_dynamic_op=True,
        maximum_cached_engines=1
    )
    
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=pd_model_path,
        conversion_params=conversion_params
    )

    converter.convert()
    converter.save(SAVE_PATH)
    
    return SAVE_PATH
  

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
    

