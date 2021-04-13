import os
import tensorflow as tf

with tf.Graph().as_default() as graph:
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

        
    tf.contrib.quantize.experimental_create_eval_graph(symmetric=True, use_qdq=True)
    saver = tf.train.Saver()
    checkpoint_dir = 'saved_results/mnist_ckpt/'
    checkpoint_name = 'model.ckpt'
    #checkpoint_dir = 'test_no_dense/'
    #checkpoint_name = 'model.ckpt'
    frozen_dir = "saved_results/"
    frozen_filename = "frozen_graph.pb"
        
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_dir + checkpoint_name)
        graph_def = graph.as_graph_def()
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, [y.op.name])
    tf.io.write_graph(frozen_graph_def,
                      os.path.dirname(frozen_dir),
                      os.path.basename(frozen_filename),
                      as_text=False)