import os
import tensorflow as tf
import tensorrt as trt
import argparse
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import common
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/MNIST_data/")
print('training info:')
print(mnist.train.images.shape, mnist.train.labels.shape)
print('testing info:')
print(mnist.test.images.shape, mnist.test.labels.shape)
print('val info:')
print(mnist.validation.images.shape, mnist.validation.labels.shape)

TRT_DYNAMIC_DIM = -1


class HostDeviceMem(object):
    r""" Simple helper data class that's a little nicer to use than a 2-tuple.
    """

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine: trt.ICudaEngine, batch_size: int):
    print('Allocating buffers ...')

    inputs = []
    outputs = []
    dbindings = []

    stream = cuda.Stream()

    for binding in engine:
        size = batch_size * abs(trt.volume(engine.get_binding_shape(binding)))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        dbindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, dbindings, stream


def infer(engine_path, batch_size, input_images, input_labels, verbose=False):
    if verbose:
        logger = trt.Logger(trt.Logger.VERBOSE)
    else:
        logger = trt.Logger(trt.Logger.INFO)

    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

        def override_shape(shape, batch_size):
            return tuple([batch_size if dim == TRT_DYNAMIC_DIM else dim for dim in shape])

        # Allocate buffers and create a CUDA stream.
        inputs, outputs, bindings, stream = allocate_buffers(engine, 1)

        with engine.create_execution_context() as context:

            # Resolve dynamic shapes in the context
            for binding in engine:
                binding_idx = engine.get_binding_index(binding)
                shape = engine.get_binding_shape(binding_idx)
                if engine.binding_is_input(binding_idx):
                    if TRT_DYNAMIC_DIM in shape:
                        shape = override_shape(shape, batch_size)
                    print('!!!!!!!!!!!!!!!!!!', binding_idx, shape)
                    context.set_binding_shape(binding_idx, shape)

            num_correct = 0
            num_total = 0
            batch_num = 0

            for start_idx in range(0, input_images.shape[0], batch_size):
                batch_num += 1

                # If the number of images in the test set is not divisible by the batch size, the last batch will be smaller.
                # This logic is used for handling that case.
                end_idx = min(start_idx + batch_size, input_images.shape[0])

                effective_batch_size = end_idx - start_idx

                # Do inference for every batch.
                # print(input_images[start_idx:start_idx + effective_batch_size, 0, :, :])

                inputs[0].host = input_images[start_idx]

                cuda.memcpy_htod(inputs[0].device, inputs[0].host)
                context.execute(batch_size, bindings)
                out = outputs[0]
                cuda.memcpy_dtoh(out.host, out.device)
                softmax_output = np.array(out.host)
                if batch_num % 1000 == 0:
                    print("Validating batch {:}".format(batch_num))
                    print(softmax_output)

                # Use argmax to get predictions and then check accuracy
                # preds = np.argmax(softmax_output.reshape(batch_size, 10)[0:effective_batch_size], axis=1)
                # labels = input_labels[start_idx:start_idx + effective_batch_size]
                # num_total += effective_batch_size
                # num_correct += np.count_nonzero(np.equal(preds, labels))

            # percent_correct = 100 * num_correct / float(num_total)
            # print("Total Accuracy: {:}%".format(percent_correct))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run inference on TensorRT engines for MNIST-based Classification models.')
    parser.add_argument('-e', '--engine', type=str, required=True, help='Path to RN50 TensorRT engine')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help="Batch size of inputs")
    parser.add_argument('-v', '--verbose', action='store_true', help="Flag to enable verbose loggin")
    args = parser.parse_args()

    # preprocessing
    input_images = mnist.test.images.reshape(-1, 28, 28, 1).transpose((0, 3, 1, 2))
    # print(input_images)
    input_labels = mnist.test.labels

    infer(args.engine, args.batch_size, input_images, input_labels, args.verbose)
