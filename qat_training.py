import os
import numpy as np
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
    y_ = tf.placeholder(tf.float32, [None, 10], name='output_0')


    def network(inputs):
        model = tf.layers.conv2d(inputs, filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                                 data_format='channels_first')  # 28x28
        model = tf.layers.conv2d(model, filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                                 data_format='channels_first')  # 28x28
        model = tf.layers.max_pooling2d(model, pool_size=(2, 2), strides=2, data_format='channels_first')  # 14x14

        model = tf.layers.conv2d(model, filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                                 data_format='channels_first')  # 14x14
        model = tf.layers.conv2d(model, filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                                 data_format='channels_first')  # 14x14
        model = tf.layers.max_pooling2d(model, pool_size=(2, 2), strides=2, data_format='channels_first')  # 7x7

        model = tf.layers.conv2d(model, filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                                 data_format='channels_first')  # 7x7
        model = tf.layers.conv2d(model, filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                                 data_format='channels_first')  # 7x7
        model = tf.layers.max_pooling2d(model, pool_size=(2, 2), strides=1, data_format='channels_first')  # 6X6

        model = tf.layers.conv2d(model, filters=512, kernel_size=(3, 3), padding='same', activation='relu',
                                 data_format='channels_first')  # 6X6
        model = tf.layers.conv2d(model, filters=512, kernel_size=(3, 3), padding='same', activation='relu',
                                 data_format='channels_first')  # 6X6
        model = tf.layers.max_pooling2d(model, pool_size=(2, 2), strides=2, data_format='channels_first')  # 3X3

        model = tf.layers.conv2d(model, filters=54, kernel_size=(3, 3), padding='same', activation='relu',
                                 data_format='channels_first')  # 3X3
        logits = tf.layers.conv2d(model, filters=10, kernel_size=(3, 3), activation='relu',
                                  data_format='channels_first', name='output_embeddings')

        # logits = tf.squeeze(logits, axis=[-2, -1])
        return logits


    logits = network(x)

    probs = tf.nn.softmax(logits, name='softmax', axis=1)
    logits = tf.squeeze(logits, axis=[-2, -1])
    # y = tf.argmax(logits, axis=1)

loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y_)
accuracy_op = tf.metrics.accuracy(labels=tf.argmax(y_, axis=1), predictions=tf.argmax(logits, axis=1))[1]

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(logits),reduction_indices=[1]))
# correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(logits, axis=1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.contrib.quantize.experimental_create_training_graph(tf.get_default_graph(), symmetric=True, use_qdq=True,
                                                       quant_delay=4500)

global_steps = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.01, global_steps, 100, 0.9, staircase=True)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_steps)
# train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

saver = tf.train.Saver(max_to_keep=3)
checkpoint_dir = 'saved_results/mnist_ckpt/'

with tf.Session() as sess:
    sess.run(init)
    for i in range(5000):
        mnist_images, batch_ys = mnist.train.next_batch(256)
        batch_xs = np.array(mnist_images * 255, dtype=np.uint8)
        train_loss, _, current_learning_rate = sess.run([loss, train_op, learning_rate],
                                                        {x: batch_xs.reshape(-1, 28, 28, 1).transpose((0, 3, 1, 2)),
                                                         y_: batch_ys.reshape(-1, 10)})
        saver.save(sess, checkpoint_dir + 'model.ckpt')
        print(train_loss)
        if (i % 100 == 0):
            print('current_learning_rate:', current_learning_rate)
            test_accuracy = sess.run(accuracy_op, {x: batch_xs.reshape(-1, 28, 28, 1).transpose((0, 3, 1, 2)),
                                                   y_: batch_ys.reshape(-1, 10)})
            print("Step=%d, Train loss=%.4f,[Test accuracy=%.2f]" % (i, train_loss, test_accuracy))

    print('acc is:')
    test_images = mnist.validation.images.reshape(-1, 28, 28, 1).transpose((0, 3, 1, 2))
    test_images = np.array(test_images * 255, dtype=np.uint8)
    print(accuracy_op.eval({x: test_images, y_: mnist.validation.labels.reshape(-1, 10)}))
    sess.close()

