import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

test = np.zeros((64,32,32,3))  # Reshape to (batch_size, channels, height, width)
print(test.nbytes)


#input_ptr = cuda.mem_alloc(test.nbytes)
#cuda.memcpy_htod(input_ptr, test)


