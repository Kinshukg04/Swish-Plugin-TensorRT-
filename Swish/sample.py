from random import randint
from PIL import Image
import numpy as np
import ctypes
import time
import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelData(object):
    MODEL_FILE = "lenet5.uff"
    INPUT_NAME ="input_1"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "dense_1/Softmax"

def build_engine(model_file):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = common.GiB(1)
        # Parse the Uff Network
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(model_file, network)
        # Build and return an engine.
        return builder.build_cuda_engine(network)

# Loads a test case into the provided pagelocked_buffer.
def load_normalized_test_case(data_paths, pagelocked_buffer, case_num=randint(0, 9)):
    [test_case_path] = common.locate_files(data_paths, [str(case_num) + ".pgm"])
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    img = np.array(Image.open(test_case_path)).ravel()
    np.copyto(pagelocked_buffer, 1.0 - img / 255.0)
    return case_num


def main():
	x = "/home/dgxuser125/rt-kennan/Swish3/build/libswish.so"
	ctypes.CDLL(x)
	data_paths, _ = common.find_sample_data(description="Runs an MNIST network using a UFF model file", subfolder="mnist")
	model_path = os.environ.get("MODEL_PATH") or os.path.join(os.path.dirname(__file__), "models")
	model_file = os.path.join(model_path, ModelData.MODEL_FILE)
	with build_engine(model_file) as engine:
		inputs, outputs, bindings, stream = common.allocate_buffers(engine)
		with engine.create_execution_context() as context:
			# # Start measuring time
			inference_start_time = time.time()
			for i in range(1000):
				case_num = load_normalized_test_case(data_paths, pagelocked_buffer=inputs[0].host)
				[output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
				pred = np.argmax(output)
				# print("Test Case: " + str(case_num))
				# print("Prediction: " + str(pred))
			end_time = time.time()
			print("time taken for one input with tenosrrt: ",(end_time-inference_start_time)/1000)
if __name__ == '__main__':
    main()
