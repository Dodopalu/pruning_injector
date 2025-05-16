# check and configure GPU 

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"avaliable GPU: {physical_devices}")
    #tf.config.experimental.set_memory_growth(physical_devices[0], True) 
else:
    print("ERROR: No GPU available, using CPU instead.")


from tensorflow import keras
from datasets.loaderCIFAR10 import load

"""
100 inferences on a 32 batch dataset
"""

# load model
#PATH = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20_pruned_50pct_20250514_185607.tflite"
PATH = "/Users/domenicopalumbo/keras_weight_injector_data/models/CIFAR10/ResNet20.keras"


model = keras.models.load_model(PATH)
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )

# load dataset 
(train_dataset, test_dataset) = load()

test_dataset = test_dataset.batch(32)



# warm-up (firts inference is generally slower)
for x in test_dataset.take(1):
    x = x[0]
    model.predict(x, verbose=0)


#=================time misuration==================#
import time

print("Starting 100 inferences...")

# Misura il tempo totale per tutte le 100 inferenze
start_time = time.time()

for batch in test_dataset:
    batch = batch[0]
    _ = model.predict(batch, verbose=0)
    
end_time = time.time()

# Calcola e mostra il tempo totale
total_time = end_time - start_time
print(f"\n===== RESULT =====")
print(f"Total time: {total_time:.4f} s")
print(f"Device: {'GPU' if physical_devices else 'CPU'}")
print("====================")