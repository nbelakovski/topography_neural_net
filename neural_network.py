# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep classifier using convolutional layers.
"""

import argparse
import sys
import tempfile
import glymur
import pickle
import numpy

import tensorflow as tf
import os
from data.subsample_matrix import subsample_matrix

FLAGS = None

# TODO: Change directory imports to data/completed. First let's get a fuckton of data there!

output_size = 28*8
crop_size = 900


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    #with tf.name_scope('reshape'):
        #x_image = tf.reshape(x, [-1, crop_size, crop_size, 3])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        w_conv1 = weight_variable([5, 5, 3, 64])
        b_conv1 = bias_variable([64])
        temp = conv2d(x, w_conv1)
        h_conv1 = tf.nn.relu(tf.add(temp, b_conv1))
        tf.summary.histogram('histogram', h_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_3x3(h_conv1)

    # Second convolutional layer -- maps 64 feature maps to 64.
    with tf.name_scope('conv2'):
        w_conv2 = weight_variable([5, 5, 64, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_3x3(h_conv2)

    # Third convolutional layer -- maps 64 feature maps to 32.
    with tf.name_scope('conv3'):
        w_conv3 = weight_variable([5, 5, 64, 32])
        b_conv3 = bias_variable([32])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)

    # Third pooling layer.
    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)

    # Fourth convolutional layer -- maps 32 feature maps to 16.
    with tf.name_scope('conv4'):
        w_conv4 = weight_variable([5, 5, 32, 16])
        b_conv4 = bias_variable([16])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, w_conv4) + b_conv4)

    # Fourth pooling layer.
    with tf.name_scope('pool4'):
        h_pool4 = max_pool_2x2(h_conv4)

    final_pool_dim = h_pool4.shape.dims[1].value
    with tf.name_scope('deconv1'):
        stride1 = 2
        w_deconv1 = weight_variable([7, 7, 8, 16])
        out_shape = [3, final_pool_dim * stride1, final_pool_dim * stride1, 8]
        h_deconv1 = tf.nn.relu(deconv2d(h_pool4, w_deconv1, out_shape, [1, stride1, stride1, 1]))

    with tf.name_scope('deconv2'):
        stride2 = 2
        w_deconv2 = weight_variable([7, 7, 4, 8])
        out_shape = [3, final_pool_dim * stride1 * stride2, final_pool_dim * stride1 * stride2, 4]
        h_deconv2 = tf.nn.relu(deconv2d(h_deconv1, w_deconv2, out_shape, [1, stride2, stride2, 1]))

    with tf.name_scope('deconv3'):
        stride3 = 2
        w_deconv3 = weight_variable([7, 7, 1, 4])
        out_shape = [3, final_pool_dim * stride1 * stride2 * stride3, final_pool_dim * stride1 * stride2 * stride3, 1]
        assert(final_pool_dim * stride1 * stride2 * stride3 == output_size)
        h_deconv3 = tf.nn.relu(deconv2d(h_deconv2, w_deconv3, out_shape, [1, stride3, stride3, 1]))
        # output = tf.constant(0.1, shape=[1, output_size, output_size, 8])
        # expected_l = conv2d(output, w_deconv, strides=[1,7,7,1])
    # pass
    # Fully connected layer 1 -- some downsampling means we have a new resolution
    # number_of_fully_connected_neurons = 1000
    # with tf.name_scope('fc1'):
    #     total_shape = 1
    #     for dim in h_pool4.shape.dims:
    #         if dim.value:
    #             total_shape *= dim.value
    #     w_fc1 = weight_variable([total_shape, number_of_fully_connected_neurons])
    #     b_fc1 = bias_variable([number_of_fully_connected_neurons])
    #
    #     h_pool4_flat = tf.reshape(h_pool4, [-1, total_shape])
    #     h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, w_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        # with tf.name_scope('dropout'):
        #   keep_prob = tf.placeholder(tf.float32)
        #   h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Map the 700x700 features to a output_sizexoutput_size feature which will be the
        # with tf.name_scope('fc2'):
        #   w_fc2 = weight_variable([1024, 10])
        #   b_fc2 = bias_variable([10])

        # y_conv = tf.reshape(h_fc1, [-1, output_size, output_size], name="final_op")
    with tf.name_scope('output'):
    #     w_output = weight_variable([number_of_fully_connected_neurons, output_size * output_size])
    #     b_output = bias_variable([output_size * output_size])
    #     h_output = tf.nn.relu(tf.matmul(h_fc1, w_output) + b_output)
        y_conv = tf.reshape(h_deconv3, [-1, output_size, output_size], name="final_op")
    return y_conv  # , keep_prob


def deconv2d(x, w, output_shape, strides=[1, 1, 1, 1]):
    """deconv2d returns a 2d deconvolution layer with full stride."""
    return tf.nn.conv2d_transpose(x, w, output_shape, strides=strides, padding='SAME')


def conv2d(x, w, strides=[1, 1, 1, 1]):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, w, strides=strides, padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def max_pool_3x3(x):
    """max_pool_3x3 downsamples a feature map by 3X."""
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 3, 3, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def import_pickle_files(directories):
    topography_data = []
    for i, directory in enumerate(directories):
        pickle_filename = [x for x in os.listdir(os.path.join('data', 'completed', directory)) if x[-6:] == 'pickle'][0]
        data = pickle.load(open(os.path.join('data', 'completed', directory, pickle_filename), 'rb'))
        topography_data.append(subsample_matrix(data, output_size))
        del data
        if i % 25 == 0:
            print("Processed", i, "pickle files")
    return topography_data


def main(_):
    # Create the model
    x = tf.placeholder(tf.float32, [None, 1008, 990, 3], name="input")

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, output_size, output_size])

    # Build the graph for the deep net
    y_conv = deepnn(x)

    with tf.name_scope('loss'):
        mean_squared_e = tf.losses.mean_squared_error(labels=y_,
                                                      predictions=y_conv)
    # mean_squared_e = tf.reduce_mean(mean_squared_e)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(mean_squared_e)

    # with tf.name_scope('accuracy'):
    #     correct_prediction = tf.squared_difference(y_conv, y_)
    #     # correct_prediction = tf.cast(correct_prediction, tf.float32)
    # accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    print("Loading jp2 files...")
    data_directories = os.listdir('data/completed')
    images = []
    directories_used = []
    for directory in data_directories:
        jp2_filename = os.path.join('data/completed', directory, 'cropped.jp2')
        if not os.path.exists(jp2_filename):
            continue
        jp2_file = glymur.Jp2k(jp2_filename)
        image_height = jp2_file.shape[0]
        image_width = jp2_file.shape[1]
        if image_height < 1008 or image_height > 1015:
            continue
        if image_width < 990 or image_width > 995:
            continue
        images.append(jp2_file[0:1008, 0:990, :])
        directories_used.append(directory)
    print("Loading jp2 files...done. Got", len(directories_used), "images")
    sys.stdout.flush()

    print("Loading pickle files...")
    subsampled_topography = import_pickle_files(directories_used)
    print("Loading pickle files...done")

    # Now we should split the data into training/test set
    total_data = len(directories_used)
    split = 0.9
    training_data_total = int(split * total_data)
    training_images = images[:training_data_total]
    test_images = images[training_data_total:]
    training_topo = subsampled_topography[:training_data_total]
    test_topo = subsampled_topography[training_data_total:]

    saver = tf.train.Saver()
    model_directory = "tnn_model"
    model_name = "tnn"
    model_path = os.path.join(model_directory, model_name)
    batch_size = 3
    total_batches = int(len(training_images) / batch_size)  # if it's not a perfect multiple, we'll leave out a few images

    # In attempting to deal with GPU issues, I will try to get tensorflow to only allocate gpu memory as needed, instead
    # of having it allocate all the GPU memory, which is what it does by default
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    merged = tf.summary.merge_all()

    with tf.Session(config=config) as sess:
        print("Initializing global variables...")
        sess.run(tf.global_variables_initializer())
        print("Initializing global variables...done")
        # TODO: Rearrange this into epochs and batches
        for i in range(11000):
            print("Creating batch...")
            batch_section = i % total_batches
            batch_start = batch_section*batch_size
            batch_end = (batch_section+1)*batch_size
            print(i, batch_start, batch_end)
            batch = [training_images[batch_start:batch_end], training_topo[batch_start:batch_end]]
            print("Creating batch...done")
            if i > 0 and i % 50 == 0:
                print("Evaluating training accuracy:")
                train_accuracy = mean_squared_e.eval(feed_dict={
                    x: training_images[0:3], y_: training_topo[0:3]})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                saver.save(sess, model_path)
            print("Running training", i, "...")
            # train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, tmp = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1]}, options=run_options, run_metadata=run_metadata)
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            print("Running training", i, "...done")
            train_writer.add_run_metadata(run_metadata, 'step %d' % i)
            train_writer.add_summary(summary, i)
            sys.stdout.flush()

        saver.save(sess, model_path)
        print('test accuracy %g' % mean_squared_e.eval(feed_dict={
            x: test_images, y_: test_topo}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
