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


train, test = load()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
engine = load_engine(trt_runtime, "DenseNet121.plan")


test = test.batch(64).take(1)

# linearize 
test = test.as_numpy_iterator()[0]
test = test.ravel()


input_ptr = cuda.mem_alloc(test.nbytes)
cuda.memcpy_htod(input_ptr, test)

output = np.empty((64, 10), dtype=np.float32).ravel()