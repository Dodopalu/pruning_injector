from loaderCIFAR10 import load
import tensorflow as tf
import numpy as np

BASE_MODEL_TFLITE = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20.tflite"


train_dataset, test_dataset = load()
dataset = test_dataset

interpreter = tf.lite.Interpreter(model_path=BASE_MODEL_TFLITE)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepara i dati di input
input_shape = input_details[0]['shape']


correct = 0
total = 0
for img, label in dataset :
    img = tf.expand_dims(img, axis=0) # (32, 32, 3) -> (1, 32, 32, 3)
    input_data = np.array(img, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = tf.argmax(output_data, axis=-1) # (1, 10) -> (1,)
    label = tf.cast(label, tf.uint8)
    output_data = tf.cast(output_data, tf.uint8)
    if output_data == label:
        correct += 1
    total += 1

accuracy = correct / total


#input_data = np.zeros(input_shape, dtype=np.float32)
#interpreter.set_tensor(input_details[0]['index'], input_data)
#interpreter.invoke()
#output_data = interpreter.get_tensor(output_details[0]['index'])


