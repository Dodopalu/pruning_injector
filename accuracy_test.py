from loaderCIFAR10 import load
import tensorflow as tf
from tensorflow import keras
import numpy as np

def accuracy(PATH : str, test_dataset, header : str):

    model = keras.models.load_model(PATH)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    
    test_dataset = test_dataset.batch(32)
    #_, accuracy = model.evaluate(test_dataset, verbose=0) # equivalent to the for loop below

    total = 0
    correct = 0
    for x, y in test_dataset:
        y = tf.squeeze(y, axis=-1) # (32,1) -> (32,)
        predictions = model.predict(x, verbose=0)
        predicted_labels = tf.argmax(predictions, axis=1)
        predicted_labels = tf.cast(predicted_labels, tf.uint8)
        correct += tf.reduce_sum(tf.cast(predicted_labels == y, tf.int32)).numpy()
        total += len(y)
        
    accuracy = correct / total

    print(10 * "=" + header + 10 * "=")
    print(" Accuracy: {:.4f}".format(accuracy))
    print(20 * "=" + len(header) * "=")
    return accuracy

def accuracy_tfLite(PATH : str, test_dataset, header : str):
    #TODO: implemet batch inference
    interpreter = tf.lite.Interpreter(model_path=PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepara i dati di input
    input_shape = input_details[0]['shape']

    correct = 0
    total = 0
    for img, label in test_dataset :
        img = tf.expand_dims(img, axis=0)
         # (32, 32, 3) -> (1, 32, 32, 3)
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

    print(10 * "=" + header + 10 * "=")
    print(" Accuracy: {:.4f}".format(accuracy))
    print(20 * "=" + len(header) * "=")

    return accuracy


if __name__ == "__main__":
    BASE_MODEL = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20.keras"
    MAGNITUDE_PRUNED_MODEL_50 = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20_sparse_50pct.keras"
    STRUCTURAL_PRUNED_MODEL_2_4 = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20_structural_2_4.keras"
    MAGNITUDE_PRUNED_MODEL_80 = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20_sparse_80pct.keras"
    STRUCTURAL_PRUNED_MODEL_5_7 = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20_structural_5_7.keras"

    train_dataset, test_dataset = load()

    #accuracy(BASE_MODEL, test_dataset, "BASE")
    #accuracy(MAGNITUDE_PRUNED_MODEL_50, test_dataset, "MAGNITUDE 0.5")
    #accuracy(STRUCTURAL_PRUNED_MODEL_2_4, test_dataset, "STRUCTURAL 2/4")
    #accuracy(MAGNITUDE_PRUNED_MODEL_80, test_dataset, "MAGNITUDE 0.8")
    #accuracy(STRUCTURAL_PRUNED_MODEL_5_7, test_dataset, "STRUCTURAL 5/7")


    # TF_LITE
    BASE_MODEL_TFLITE = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20.tflite"
    MAGNITUDE_PRUNED_MODEL_50_TFLITE = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20_sparse_50pct.tflite"
    MAGNITUDE_PRUNED_MODEL_80_TFLITE = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20_sparse_80pct.tflite"
    STRUCTURAL_PRUNED_MODEL_5_7_TFLITE = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20_structural_5_7.tflite"
    STRUCTURAL_PRUNED_MODEL_2_4_TFLITE = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20_structural_2_4.tflite"

    accuracy_tfLite(BASE_MODEL_TFLITE, test_dataset, "BASE")
    accuracy_tfLite(MAGNITUDE_PRUNED_MODEL_50_TFLITE, test_dataset, "MAGNITUDE 0.5")
    accuracy_tfLite(STRUCTURAL_PRUNED_MODEL_2_4_TFLITE, test_dataset, "STRUCTURAL 2/4")
    accuracy_tfLite(MAGNITUDE_PRUNED_MODEL_80_TFLITE, test_dataset, "MAGNITUDE 0.8")
    accuracy_tfLite(STRUCTURAL_PRUNED_MODEL_5_7_TFLITE, test_dataset, "STRUCTURAL 5/7")
