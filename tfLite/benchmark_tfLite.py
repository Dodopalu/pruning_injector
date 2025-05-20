# check and configure GPU 
import tensorflow as tf
import time
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"avaliable GPU: {physical_devices}")
    #tf.config.experimental.set_memory_growth(physical_devices[0], True) 
else:
    print("ERROR: No GPU available, using CPU instead.")

from tensorflow import keras
from loaderCIFAR10 import load

"""
100 inferenze su batch di 32 elementi con modello TFLite
"""
# Carica il dataset
def test(header, PATH):

    num_of_inferenzes = 10000

    (train_dataset, test_dataset) = load()


    interpreter = tf.lite.Interpreter(
        model_path=PATH,
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    # tf.dataset (32, 32, 3) -> np.array (1, 32, 32, 3)
    test_batches = []
    for images, _ in test_dataset.take(num_of_inferenzes): 
        img = np.expand_dims(images.numpy(), axis=0)
        test_batches.append(img)

    # Warm-up
    interpreter.set_tensor(input_details[0]['index'], test_batches[0])
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])

    interpreter.set_tensor(input_details[0]['index'], test_batches[0])
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])

    # benchmark 
    start_time = time.time()


    for i in range(num_of_inferenzes):
        batch_idx = i % len(test_batches)
        interpreter.set_tensor(input_details[0]['index'], test_batches[batch_idx])
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        
    
    end_time = time.time()
    total_time = end_time - start_time

    print (7 * "=" + header + 7 * "=")
    print(f"Total time : {total_time:.4f} s")
    print(14*"=" + len(header)*"=")



if __name__ == "__main__":
    # Percorsi modelli
    BASE_MODEL = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20.tflite"
    MAGNITUDE_PRUNED_MODEL_50 = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20_sparse_50pct_20250514_185607.tflite"
    STRUCTURAL_PRUNED_MODEL_2_4 = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20structural_20250515-133441.tflite"
    MAGNITUDE_PRUNED_MODEL_80 = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20_sparse_80pct.tflite"
    STRUCTURAL_PRUNED_MODEL_5_7 = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20_structural_5_7.tflite"

    test("BASE", BASE_MODEL)
    test("MAGITUDE 0.5", MAGNITUDE_PRUNED_MODEL_50)
    test("STRUCTURAL 2/4", STRUCTURAL_PRUNED_MODEL_2_4)
    test("MAGNITUDE 0.8", MAGNITUDE_PRUNED_MODEL_80)
    test("STRUCTURAL 5/7", STRUCTURAL_PRUNED_MODEL_5_7)


    