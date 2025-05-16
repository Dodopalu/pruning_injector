import os
import datetime
from tensorflow import keras
import tensorflow as tf

PATH = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20.keras"
PATH = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20_sparse_50pct_20250514_185607.keras"
PATH = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20structural_20250515-133441.keras"



def convert_to_tfLite(PATH, OUTPUT_DIR):

    model = keras.models.load_model(PATH)
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model_name = os.path.basename(PATH).split('.')[0]
    filename = f"{model_name}.tflite"

    tflite_file = os.path.join(OUTPUT_DIR, filename)

    # Covert model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    converter.experimental_enable_resource_variables = True
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT,
        tf.lite.Optimize.EXPERIMENTAL_SPARSITY
                            ]

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    tflite_model = converter.convert()



    print('Saving pruned model to:', tflite_file)
    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)

    NEW_PATH = os.path.join(OUTPUT_DIR, f"{model_name}.tflite")
    return NEW_PATH


if __name__ == "__main__":

    OUTPUT_DIR = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10"
    
    BASE_MODEL = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20.keras"
    MAGNITDE_50 = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20_sparse_50pct_20250514_185607.keras"
    STRUCTURAL_2_4 = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20structural_20250515-133441.keras"

    #convert_to_tfLite(STRUCTURAL_2_4, OUTPUT_DIR)
    convert_to_tfLite(BASE_MODEL, OUTPUT_DIR)