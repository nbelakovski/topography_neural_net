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

"""A deep regressor using convolutional layers.
"""

import argparse
import numpy
import sys
import tempfile
import glymur
import tensorflow as tf
import os
from random import random
from data.subsample_matrix import subsample_matrix
from read_data import read_data

FLAGS = None

output_size = 28*8  # Current comes from size of last input layer and 3 deconv layers with stride 2
batch_size = 3


def deepnn(x):
    """deepnn builds the graph for a deep net for determining topography from image data.

  Args:
    x: an input tensor with the dimensions (N_examples, 1008, 990), where 1008x990 is the
    number of pixels in the input image.

  Returns:
    A tensor y of shape (output_size, output_size), with values equal to the topography of the input image
  """

    # First convolutional layer - maps one image to a bunch of feature maps.
    with tf.name_scope('conv1'):
        w_conv1 = weight_variable([5, 5, 3, 64])
        b_conv1 = bias_variable([64])
        temp = conv2d(x, w_conv1)
        h_conv1 = tf.nn.relu(tf.add(temp, b_conv1))
        tf.summary.histogram('histogram', h_conv1)

    # Pooling layer - downsamples.
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

    # First deconvolution layer. Upsample the last pooling layer and halve the number of feature maps
    final_pool_dim = h_pool4.shape.dims[1].value
    with tf.name_scope('deconv1'):
        stride1 = 2
        w_deconv1 = weight_variable([7, 7, 8, 16])  # height, width, out channels, in channels
        out_shape = [batch_size, final_pool_dim * stride1, final_pool_dim * stride1, 8]
        h_deconv1 = tf.nn.relu(deconv2d(h_pool4, w_deconv1, out_shape, [1, stride1, stride1, 1]))

    # Second deconvolutional layer
    with tf.name_scope('deconv2'):
        stride2 = 2
        w_deconv2 = weight_variable([7, 7, 4, 8])
        out_shape = [batch_size, final_pool_dim * stride1 * stride2, final_pool_dim * stride1 * stride2, 4]
        h_deconv2 = tf.nn.relu(deconv2d(h_deconv1, w_deconv2, out_shape, [1, stride2, stride2, 1]))

    # Third deconvolutional layer
    with tf.name_scope('deconv3'):
        stride3 = 2
        w_deconv3 = weight_variable([7, 7, 1, 4])
        out_shape = [batch_size, final_pool_dim * stride1 * stride2 * stride3,
                     final_pool_dim * stride1 * stride2 * stride3, 1]
        assert(final_pool_dim * stride1 * stride2 * stride3 == output_size)
        h_deconv3 = tf.nn.relu(deconv2d(h_deconv2, w_deconv3, out_shape, [1, stride3, stride3, 1]))

    with tf.name_scope('output'):
        # For use in the loss function, and in the inference script, rearrange the last layer to remove batch size and
        # the reference to the single channel
        y_conv = tf.reshape(h_deconv3, [-1, output_size, output_size], name="final_op")
    return y_conv


def deconv2d(x, w, output_shape, strides=None):
    """deconv2d returns a 2d deconvolution layer with full stride."""
    if strides is None:
        strides = [1, 1, 1, 1]
    return tf.nn.conv2d_transpose(x, w, output_shape, strides=strides, padding='SAME')


def conv2d(x, w, strides=None):
    """conv2d returns a 2d convolution layer with full stride."""
    if strides is None:
        strides = [1, 1, 1, 1]
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


def import_data_files(directories):
    topography_data = []
    for i, directory in enumerate(directories):
        topo_filename = [x for x in os.listdir(os.path.join(FLAGS.data_dir, 'completed', directory)) if x[-5:] == '.data'][0]
        data = read_data(os.path.join(FLAGS.data_dir,'completed', directory, topo_filename))
        data = numpy.matrix(data)
        topography_data.append(subsample_matrix(data, output_size))
        del data
        if i % 25 == 0:
            print("Processed", i, "data files")
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

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(mean_squared_e)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    print("Loading jp2 files...")
    data_directories = os.listdir(FLAGS.data_dir + '/completed')
    images = []
    directories_used = []
    for directory in data_directories:
        jp2_filename = os.path.join(FLAGS.data_dir + '/completed', directory, 'cropped.jp2')
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

    print("Loading data files...")
    subsampled_topography = import_data_files(directories_used)
    print("Loading data files...done")

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
    total_batches = int(len(training_images) / batch_size)  # if it's not a perfect multiple, we'll leave out few images

    # In attempting to deal with GPU issues, I will try to get tensorflow to only allocate gpu memory as needed, instead
    # of having it allocate all the GPU memory, which is what it does by default
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # Try importing the model and loading the last checkpoint, if possible. Otherwise initialize from scratch
        try:
            print("Loading model from disk...")
            saver.restore(sess, model_path)
            print("Loading model from disk...done")
        except:
            print("Loading model from disk failed.")
            print("Initializing global variables...")
            sess.run(tf.global_variables_initializer())
            print("Initializing global variables...done")

        for epoch in range(11000):
            print("Running epoch", epoch)
            print("Shuffling...")
            # Need to keep topo correlated to the images. So, shuffle the indices, and then create new arrays
            # based on the shuffled indices
            index_array = [x for x in range(len(training_images))]
            index_array.sort(key=lambda k: k * random())
            # Now create new arrays to store the values
            new_training_images = []
            new_training_topo = []
            for i in index_array:
                new_training_images.append(training_images[i])
                new_training_topo.append(training_topo[i])
            # Now re-assign
            training_images = new_training_images
            training_topo = new_training_topo
            # Maintenance of correlation has been verified in debug mode with spot checks
            print("Shuffling...done")
            for batch_number in range(total_batches):
                print("Creating batch...")
                batch_start = batch_number*batch_size
                batch_end = (batch_number+1)*batch_size
                print(epoch, batch_number, batch_start, batch_end)
                batch = [training_images[batch_start:batch_end], training_topo[batch_start:batch_end]]
                print("Creating batch...done")
                print("Running training", epoch, "-", batch_number, "...")
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                sess.run([train_step], feed_dict={x: batch[0], y_: batch[1]}, options=run_options,
                         run_metadata=run_metadata)
                print("Running training", epoch, "-", batch_number, "...done")
                train_writer.add_run_metadata(run_metadata, 'step %d - %d' % (epoch, batch_number))
                sys.stdout.flush()
            print("Evaluating training accuracy after:")
            train_accuracy = mean_squared_e.eval(feed_dict={
                x: test_images[0:batch_size], y_: test_topo[0:batch_size]})
            print('epoch %d, training accuracy %g' % (epoch, train_accuracy))
            saver.save(sess, model_path)

        saver.save(sess, model_path)
        print('test accuracy %g' % mean_squared_e.eval(feed_dict={
            x: test_images, y_: test_topo}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/home/docker/data',
                        help='Directory for input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
