import keras
import onnx
import tf2onnx
import os


PATH = "DenseNet121_structural_5_7.keras"


model = keras.models.load_model(PATH)
onnx_model, _ = tf2onnx.convert.from_keras(model, output_path="DenseNet121_structural_5_7.onnx")
onnx.save_model(onnx_model, "DenseNet121_structural_5_7.onnx")


