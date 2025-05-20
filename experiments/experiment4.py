import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt

path = "./models/flatten_Vgg11_bn.keras"
model = tf.keras.models.load_model(path)
model.save("my_model")


converter = trt.TrtGraphConverterV2(
    input_saved_model_dir="./my_model",
    precision_mode=trt.TrtPrecisionMode.FP32
    )


converter.convert()

# Save the model to the disk 
converter.save(output_saved_model_dir="./models_tensorRT")