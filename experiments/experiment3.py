import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt

def get_model():
    # Create a simple model.
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

model = get_model()


test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
model.fit(test_input, test_target)

model.save("my_model.keras")
md = keras.models.load_model("my_model.keras")



#converter = trt.TrtGraphConverterV2(
#    input_saved_model_dir="./my_model",
#    precision_mode=trt.TrtPrecisionMode.FP32
#    )


#converter.convert()

# Save the model to the disk 
#converter.save(output_saved_model_dir="./models_tensorRT")