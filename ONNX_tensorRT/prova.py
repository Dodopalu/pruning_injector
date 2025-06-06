import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

test = np.zeros((64,32,32,3))  # Reshape to (batch_size, channels, height, width)
print(test.nbytes)


input_ptr = cuda.mem_alloc(test.nbytes)
cuda.memcpy_htod(input_ptr, test)


import tensorrt as trt

def load_engine(trt_runtime, plan_path) -> trt.ICudaEngine:
   with open(plan_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
engine = load_engine(trt_runtime, "DenseNet121.plan")

#engine = load_engine(trt_runtime, "DenseNet121.plan")



