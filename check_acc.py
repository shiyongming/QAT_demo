import tensorrt as trt
import tensorflow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/MNIST_data/", one_hot=True)
print('training info:')
print(mnist.train.images.shape, mnist.train.labels.shape)
print('testing info:')
print(mnist.test.images.shape, mnist.test.labels.shape)
print('val info:')
print(mnist.validation.images.shape, mnist.validation.labels.shape)

test_images = mnist.test.images.reshape(-1, 28, 28, 1).transpose((0, 3, 1, 2))
test_labels = mnist.test.labels

def load_engine(engine_filepath, trt_logger):
    with open(engine_filepath, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        print('engine is loaded!!!!!')
    return engine

def check_accuracy(context, batch_size, test_images, test_labels):
    inputs, outputs, bindings, stream = common.allocate_buffers(context.engine)
    num_correct = 0
    num_total = 0
    batch_num = 0
    print(test_images.shape[0],'!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    for start_idx in range(0, test_images.shape[0], batch_size):
        batch_num += 1
        if batch_num % 10 == 0:
            print("Validating batch {:}".format(batch_num))
        end_idx = min(start_idx + batch_size, test_set.shape[0])
        effective_batch_size = end_idx - start_idx
        inputs[0].host = test_images[start_idx:start_idx + effective_batch_size, :, :, :]
        [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=effective_batch_size)
        print(output)


def main():
    engine_file_path = "mnist_qat.engine"
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    with load_engine(engine_file_path, TRT_LOGGER) as engine, engine.create_execution_context() as context:
        check_accuracy(context, batch_size, test_images=test_images, test_labels=test_labels)
        