import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import load_engine
import tensorrt as trt
import tensorflow as tf
import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from onnx import ModelProto

os.environ["CUDA_VISIBLE_DEVICES"]="0"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)



'''
Loading file for CIFAR10
'''
from tensorflow import keras
import tensorflow as tf

# to be loaded 
def load() -> tuple[tf.data.Dataset, tf.data.Dataset]:

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()


    test_images = tf.data.Dataset.from_tensor_slices(test_images)
    test_labels = tf.data.Dataset.from_tensor_slices(test_labels)
    train_images = tf.data.Dataset.from_tensor_slices(train_images)
    train_labels = tf.data.Dataset.from_tensor_slices(train_labels)

    def preprocess_img(img : tf.Tensor) -> tf.Tensor:
        mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
        std = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)

        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = (img - mean) / std
        return img
    
    train_images = train_images.map(preprocess_img)
    test_images = test_images.map(preprocess_img)

    # trasform into tensor
    train_dataser = tf.data.Dataset.zip((train_images, train_labels))
    validation_dataset = tf.data.Dataset.zip((test_images, test_labels))

    return train_dataser, validation_dataset






def build_engine(onnx_path, shape, max_batch_size=64) -> trt.ICudaEngine:

   """
   This is the function to create the TensorRT engine
   Args:
      onnx_path : Path to onnx_file. 
      shape : Shape of the input of the ONNX file. 
  """
   with (
         trt.Builder(TRT_LOGGER) as builder, 
         builder.create_network(1) as network, 
         builder.create_builder_config() as config, 
         trt.OnnxParser(network, TRT_LOGGER) as parser
         ):
       

       builder.max_batch_size = max_batch_size

       config.set_flag(trt.BuilderFlag.TF32)
       config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)       
       config.max_workspace_size = (1 << 33)

       with open(onnx_path, 'rb') as model:
           parser.parse(model.read())
        
       network.get_input(0).shape = shape
       engine = builder.build_engine(network, config)
       
       return engine

def save_engine(engine : trt.ICudaEngine, file_name):
   buf = engine.serialize()
   with open(file_name, 'wb') as f:
       f.write(buf)

def load_engine(trt_runtime, plan_path) -> trt.ICudaEngine:
   with open(plan_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine

def load_data_to_gpu(dt : tf.data.Dataset, batch_size : int, context) -> tuple[list, list]:
    dt = dt.batch(64)
    dt_np = list(dt.as_numpy_iterator()) # (data, label), (data, label)...


    # create the linearized dataset
    input_linearized = []
    labels = []
    for data, label in dt_np:
        input_linearized.append(data.ravel())
        labels.append(label)

    # allocate dataset in GPU
    input_gpu_ptrs = []
    for i in range(len(input_linearized)):
        ptr = cuda.mem_alloc(input_linearized[i].nbytes)
        cuda.memcpy_htod(ptr, input_linearized[i])
        input_gpu_ptrs.append(ptr)


    output_gpu_ptrs = [np.zeros((batch_size, 10), dtype=np.float32)] * len(input_gpu_ptrs)

    return input_gpu_ptrs, output_gpu_ptrs

def inference(engine : trt.ICudaEngine , list_input_ptr : list, list_output_ptr : list, batch_size : int, context):
    
    if len(list_input_ptr) != len(list_output_ptr):
        raise ValueError("Number of input pointers does not match the engine's input count.")

    time_0 = time.localtime()


    for i in range(len(list_input_ptr)):

        binding = list()
        binding.append(list_input_ptr[i])
        binding.append(list_output_ptr[i])

        context.execute(
            batch_size, 
            bindings=binding
            )

    time_1 = time.localtime()


    #evaluate output
    res = []
    for output_ptr in list_output_ptr:
        output_data = np.empty((batch_size, 10), dtype=np.float32)
        cuda.memcpy_dtoh(output_data, output_ptr)
        res.append(output_data)
    
    print("INFO on output data")
    print(f"Output shape: {res[0].shape}")
    print(f"Output data: {res[0]}")
    print(f"Output data type: {res[0].dtype}")
    print(f"Output length: {len(res)}")

    # save np array
    saved = np.concatenate(res, axis=0)
    name = time.strftime("%Y%m%d-%H%M%S")
    print(f"Saving output data to {name}.npy")
    np.save(f"{name}.npy", saved)

    timestamp0 = time.mktime(time_0)
    timestamp1 = time.mktime(time_1)

    return timestamp1 - timestamp0


#--------------------------------------------------------------------#
def total_experiment():
    name = "DenseNet121" 
    #name = "DenseNet121_structural_2_4"
    #name = "DenseNet121_structural_5_7"
    engine_file = f"./{name}.plan"
    onnx_path = f"ONNX_tensorRT/{name}.onnx"
    batch_size = 64
    height = 32
    width = 32

    model = ModelProto()
    with open(onnx_path, "rb") as f:
        model.ParseFromString(f.read())

    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value

    print(f"Input shape from ONNX: {d0}, {d1}, {d2}")

    shape = [batch_size , height, width, 3]
    engine = build_engine(onnx_path, shape= shape, max_batch_size=batch_size)
    save_engine(engine, engine_file)
    print(shape)


    train, test = load()
    dt = test.take(100)
    input_ptr_list, output_ptr_list = load_data_to_gpu(dt, batch_size=64, engine=engine)
    print(f"Loaded {len(input_ptr_list)} input tensors to GPU.")

    warmup = inference(
        engine=engine, 
        list_input_ptr=input_ptr_list[:10], 
        list_output_ptr=output_ptr_list[:10], 
        batch_size=64
    )
    print(f"Warmup completed.")

    print("Starting inference...")
    time = inference(
        engine=engine, 
        list_input_ptr=input_ptr_list, 
        list_output_ptr=output_ptr_list, 
        batch_size=64
    )
    print(f"Inference time: {time:.4f} seconds")

def build_and_save_engine():
    name = "DenseNet121" 
    #name = "DenseNet121_structural_2_4"
    #name = "DenseNet121_structural_5_7"
    engine_file = f"./{name}.plan"
    onnx_path = f"ONNX_tensorRT/{name}.onnx"
    batch_size = 64
    height = 32
    width = 32

    model = ModelProto()
    with open(onnx_path, "rb") as f:
        model.ParseFromString(f.read())

    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value

    shape = [batch_size , height, width, 3]
    engine = build_engine(onnx_path, shape= shape, max_batch_size=batch_size)
    save_engine(engine, engine_file)
    print(shape)

def load_and_infer():
    name = "DenseNet121"
    #name = "DenseNet121_structural_2_4"
    #name = "DenseNet121_structural_5_7"
    engine_serialized = f"./{name}.plan"

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    engine = load_engine(trt_runtime, engine_serialized)


    with engine.create_execution_context() as context:

        train, test = load()
        dt = test

        input_ptr_list, output_ptr_list = load_data_to_gpu(dt, batch_size=64, context=context)




        print(f"Loaded {len(input_ptr_list)} input tensors to GPU.")

        warmup = inference(
            engine=engine, 
            list_input_ptr=input_ptr_list[:10], 
            list_output_ptr=output_ptr_list[:10], 
            batch_size=64,
            context=context
        )
        print(f"Warmup completed.")

        print("Starting inference...")
        time = inference(
            engine=engine, 
            list_input_ptr=input_ptr_list, 
            list_output_ptr=output_ptr_list, 
            batch_size=64,
            context=context
        )
        print(f"Inference time: {time:.4f} seconds")


load_and_infer()
