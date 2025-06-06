import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

test = np.zeros((64,32,32,3))  # Reshape to (batch_size, channels, height, width)
test[0, 0, 0, 0] = 1.0  # Example of setting a value in the array
print(test.nbytes)


input_ptr = cuda.mem_alloc(test.nbytes)
cuda.memcpy_htod(input_ptr, test)


out = np.array(np.zeros((64, 10)), dtype=np.float32)  # Assuming output shape is (batch_size, num_classes)
cuda.memcpy_dtoh(input_ptr, out)

print(out.shape)



