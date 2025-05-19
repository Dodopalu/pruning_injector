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


def save_model_tensorRT(model_path : str, output_dir : str) -> str :
    conversion_params = trt.TrtConversionParams(
        precision_mode=trt.TrtPrecisionMode.FP32
        )

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=BASE_MODEL,
        conversion_params=conversion_params
        )

    # Converter method used to partition and optimize TensorRT compatible segments
    def calibration():
        _, dataset = load()
        dataset = dataset.batch(10)
        for batch in dataset.take(10):
            yield [tf.constant(batch[0])]

    converter.convert(calibration_input_fn=calibration)

    # Save the model to the disk 
    converter.save(output_saved_model_dir=OUTPUT_DIR)

    model_name = model_path.split("/")[-1]
    return f"{output_dir}/{model_name}"

# accuracy test


# inference test
def inference_test(model_path):

    # Load the model
    model = tf.saved_model.load(model_path)
    infer = model.signatures['serving_default']

    # load dataset 
    (train_dataset, test_dataset) = load()
    test_dataset = test_dataset.batch(32)


    # Precaricamento dati nella GPU
    gpu_dataset = []
    for img, labels in test_dataset:
        x = img # take input not labels
        x_gpu = tf.convert_to_tensor(x, dtype=tf.float32)
        if physical_devices:
            with tf.device('/GPU:0'):
                x_gpu = tf.identity(x_gpu)
        gpu_dataset.append(x_gpu)

    # Warm-up 
    for i, batch in enumerate(gpu_dataset):
        if i < 10:
            _ = infer(tf.constant(batch))

    # Sync
    if physical_devices:
        _ = tf.random.normal([1])


    print("Starting 100 inferences...")

    start_time = time.time()

    for img, labels in gpu_dataset.take(100):
        _ = infer(batch)

    end_time = time.time()

    # Results
    total_time = end_time - start_time
    print(f"\n===== RESULT =====")
    print(f"Total time: {total_time:.4f} s")
    print("=" * 18)


if __name__ == "__main__":
    OUTPUT_DIR = "models_tensorRT"

    BASE_MODEL = "models/ResNet20.keras"
    MAGNITUDE_50 = "models/ResNet20_sparse_50.keras"
    MAGNITUDE_80 = "models/ResNet20_sparse_80.keras"
    STRUCTURAL_2_4 = "models/ResNet20_structural_2_4.keras"
    STRUCTURAL_5_7 = "models/ResNet20_structural_5_7.keras"

    # base model
    path = save_model_tensorRT(BASE_MODEL, OUTPUT_DIR)
    inference_test(path)

    # sparse pruning 50%
    #path = save_model_tensorRT(MAGNITUDE_50, OUTPUT_DIR)
    #inference_test(path)

    # sparse pruning 80%
    #path = save_model_tensorRT(MAGNITUDE_80, OUTPUT_DIR)
    #inference_test(path)

    # structural pruning 2-4
    path = save_model_tensorRT(STRUCTURAL_2_4, OUTPUT_DIR)
    inference_test(path)

    # structural pruning 5-7
    #path = save_model_tensorRT(STRUCTURAL_5_7, OUTPUT_DIR)
    #inference_test(path)
