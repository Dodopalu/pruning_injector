# check and configure GPU 
import time
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"avaliable GPU: {physical_devices}")
    #tf.config.experimental.set_memory_growth(physical_devices[0], True) 
else:
    print("ERROR: No GPU available, using CPU instead.")


from tensorflow import keras
from loaderCIFAR10 import load

"""
100 inferences on a 32 batch dataset
"""

def benchmark(model_path : str, dataset : tf.data.Dataset, batch : int = 32):

    dataset = dataset.batch(batch)

    model = keras.models.load_model(model_path)
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        )


    # Precaricamento dati nella GPU
    gpu_dataset = []
    for img, labels in dataset:
        x = img # take input not labels
        x_gpu = tf.convert_to_tensor(x, dtype=tf.float32)
        if physical_devices:
            with tf.device('/GPU:0'):
                x_gpu = tf.identity(x_gpu)
        gpu_dataset.append(x_gpu)

    #gpu_dataset = test_dataset.map(lambda x, y: x) # not to send to GPU

    # Warm-up 
    for i, batch in enumerate(gpu_dataset):
        if i < 10:
            _ = model.predict(batch, verbose=0)

    # Sync
    if physical_devices:
        _ = tf.random.normal([1])

    print("Starting 100 inferences...")

    start_time = time.time()

    for batch in gpu_dataset:
        _ = model.predict(batch, verbose=0)
        
    end_time = time.time()

    # results
    total_time = end_time - start_time
    print(f"\n===== RESULT =====")
    print(f"Total time: {total_time:.4f} s")
    print("====================")


def benchmark_trt(model_path: str, dataset: tf.data.Dataset, batch: int = 32):

    dataset = dataset.batch(batch).take(1)
 
    trt_model = tf.saved_model.load(model_path)
    infer_fn = trt_model.signatures['serving_default']
    input_name = list(infer_fn.structured_input_signature[1].keys())[0]

    print(f"input name: {input_name}")
    
    gpu_dataset = []
    for img, labels in dataset:
        x = img  # take input not labels
        x_gpu = tf.convert_to_tensor(x, dtype=tf.float32)
        if physical_devices:
            with tf.device('/GPU:0'):
                x_gpu = tf.identity(x_gpu)
        gpu_dataset.append(x_gpu)
    
    # Warm-up 
    for i, batch in enumerate(gpu_dataset):
        if i < 10:
            _ = infer_fn(**{input_name: batch})
    
    # Sync
    if physical_devices:
        _ = tf.random.normal([1])
    
    print("Starting 100 inferences...")
    
    start_time = time.time()
    
    for batch in gpu_dataset:
        _ = infer_fn(**{input_name: batch})
        
    end_time = time.time()
    
    # results
    total_time = end_time - start_time
    print(f"\n===== RESULT =====")
    print(f"Total time: {total_time:.4f} s")
    print("====================")