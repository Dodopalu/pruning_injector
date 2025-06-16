import tensorflow as tf
from tensorflow import keras
import tf2onnx
import onnx
import numpy as np


# 1. Carica il tuo modello Keras
model = keras.models.load_model("ResNet20_2_4.keras")

input_shape = model.input_shape[1:]  # Forma senza la dimensione del batch

# 4. Crea una specifica di input con batch size dinamico
input_signature = [tf.TensorSpec([None, 32, 32, 3], tf.float32, name='input')]

# 5. Converti il modello in ONNX
output_path = "ResNet20_2_4.onnx"
model_proto, _ = tf2onnx.convert.from_keras(
    model, 
    #input_signature=input_signature,
    #opset=11,
    output_path=output_path,
)

model = onnx.load(output_path)
model.graph.output[0].name = "output"



onnx.save(model, output_path)
