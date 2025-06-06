import pycuda.driver as cuda
import pycuda.autoinit

alloc = cuda.mem_alloc(10)
