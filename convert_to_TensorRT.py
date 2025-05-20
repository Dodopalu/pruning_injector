import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

VGG = "./models/flatten_Vgg11_bn.keras"
VGG_DIR = "./models_pd/flatten_Vgg11_bn"
VGG_OUT = "./models_tensorRT/latten_Vgg11_bn"


model = tf.keras.models.load_model(VGG)
model.save(VGG_DIR)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=VGG_DIR,
    precision_mode=trt.TrtPrecisionMode.FP32
    )
converter.convert()
converter.save(output_saved_model_dir=VGG_OUT)