import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from datasets.loaderCIFAR10 import load
import numpy as np


test, train = load()
dt = test.batch(64)
dt_np = list(dt.as_numpy_iterator()) # (data, label), (data, label)...


# create the linearized dataset
input_linearized = []
labels = []
for data, label in dt_np:
    input_linearized.append(data.ravel())
    labels.append(label)

# delinearize the dataset
flat_array = np.array(input_linearized[0])
original = flat_array.reshape(64, 32, 32, 3)

print(
    np.all(original == dt_np[0][0]) and             # all element are equal
    (original == dt_np[0][0])[0, 0, 0, 0] == True   # equal to true
    )


