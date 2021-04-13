import os
import tensorflow as tf
print(tf.__version__)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/MNIST_data/", one_hot=True)
print('training info:')
print(mnist.train.images.shape, mnist.train.labels.shape)
print('testing info:')
print(mnist.test.images.shape, mnist.test.labels.shape)
print('val info:')
print(mnist.validation.images.shape, mnist.validation.labels.shape)


with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, [None, 1, 28, 28], name='input_0')
    def network(inputs):
        model = tf.layers.conv2d(inputs, filters=16, kernel_size=(3, 3), padding='same', data_format='channels_first') #28x28
        model = tf.layers.conv2d(model, filters=32, kernel_size=(3, 3), padding='same', activation='relu', data_format='channels_first') #28x28
        model = tf.layers.max_pooling2d(model, pool_size=(2, 2), strides=2, data_format='channels_first') #14x14

        model = tf.layers.conv2d(model, filters=64, kernel_size=(3, 3), padding='same', data_format='channels_first') #14x14
        model = tf.layers.conv2d(model, filters=128, kernel_size=(3, 3), padding='same', activation='relu', data_format='channels_first')#14x14
        model = tf.layers.max_pooling2d(model, pool_size=(2, 2), strides=2, data_format='channels_first') #7x7
        
        model = tf.layers.conv2d(model, filters=32, kernel_size=(3, 3), padding='same', data_format='channels_first') #7x7
        logits = tf.layers.conv2d(model, filters=10, kernel_size=(7, 7), data_format='channels_first', name='output_embeddings')

        logits = tf.squeeze(logits, axis=[-2, -1])
        return logits
    
    embeddings = network(x)
    y = tf.nn.softmax(embeddings, name='softmax')
    y_ = tf.placeholder(tf.float32, [None, 10], name='output_0')


    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

tf.contrib.quantize.experimental_create_training_graph(tf.get_default_graph(), symmetric=True, use_qdq=True, quant_delay=100)
train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

saver=tf.train.Saver(max_to_keep=3)
init = tf.global_variables_initializer()
checkpoint_dir = 'mnist_ckpt/'

with tf.Session() as sess:
    sess.run(init)

    for i in range(200):
        batch_xs, batch_ys = mnist.train.next_batch(256)
        train_step.run({x: batch_xs.reshape(-1, 28, 28, 1).transpose((0, 3, 1, 2)), y_: batch_ys.reshape(-1, 10)})
        saver.save(sess, checkpoint_dir + 'model.ckpt')
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('acc is:')
    print(accuracy.eval({x: mnist.validation.images.reshape(-1, 28, 28, 1).transpose((0, 3, 1, 2)), y_: mnist.validation.labels.reshape(-1, 10)}))
    sess.close()
    
#        graph_def = graph.as_graph_def()
#        frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, [y.op.name])
#        
#    tf.io.write_graph(frozen_graph_def,
#                      os.path.dirname('frezon/'),
#                      os.path.basename('frozon.pb'),
#                      as_text=False)

