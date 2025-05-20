#from pruning_magnitude import sparse_pruning
#from pruning_structural import structural_pruning
from accuracy_test import accuracy
import time
import tensorflow as tf
import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"avaliable GPU: {physical_devices}")
    #tf.config.experimental.set_memory_growth(physical_devices[0], True) 
else:
    print("ERROR: No GPU available, using CPU instead.")


def load():

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


model = keras.models.load_model(
    filepath="./models/CIFAR10/resnet/ResNet20.keras",
    custom_objects={},
    compile=False
    )
model.save(
    "./models/saved_model",
    )



converter = trt.TrtGraphConverterV2(
    input_saved_model_dir="./models/saved_model",
    precision_mode=trt.TrtPrecisionMode.FP32
    )


# Converter method used to partition and optimize TensorRT compatible segments
def calibration():
    _, dataset = load()
    dataset = dataset.batch(10)
    for batch in dataset.take(10):
        yield [tf.constant(batch[0])]

converter.convert()
converter.summary()

# Save the model to the disk 
converter.save(output_saved_model_dir="./models_tensorRT")
