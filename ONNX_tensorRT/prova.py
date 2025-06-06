import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import load_engine
import tensorrt as trt
import tensorflow as tf
import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from onnx import ModelProto

from experiment import load, load_engine


test = np.zeros((64,32,32,3))  # Reshape to (batch_size, channels, height, width)
print(test.nbytes)


#input_ptr = cuda.mem_alloc(test.nbytes)
#cuda.memcpy_htod(input_ptr, test)




#output = np.empty((64, 10), dtype=np.float32).ravel()

#output_ptr = cuda.mem_alloc(output.nbytes)

#bindings = [int(input_ptr), int(output_ptr)]



#with engine.create_execution_context() as context:
#    context.execute_v2(bindings=bindings)
