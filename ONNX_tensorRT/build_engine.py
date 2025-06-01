import engine as eng
import argparse
from onnx import ModelProto
import tensorrt as trt 
 
name = "model" 
engine_file = f"./{name}.plan"
onnx_path = f"/home/nikilr2/trt_tutorial/braggnn-pytorch/{name}.onnx"
batch_size = 64
height = 32
width = 32

model = ModelProto()
with open(onnx_path, "rb") as f:
    model.ParseFromString(f.read())

d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value

shape = [batch_size , height, width, 3]
engine = eng.build_engine(onnx_path, shape= shape, max_batch_size=batch_size)
eng.save_engine(engine, engine_file)
print(shape)