#!/usr/bin/python3

# Attribution: This code started from the Tensorflow deep MNIST convulational
# example, and was expanded upon from there.

"""A deep regressor using convolutional layers."""

# Python libraries (alphabetical)
import argparse
import glymur
from math import ceil, gcd
from multiprocessing import Queue, Process, Manager
import numpy as np
import os
import queue
from random import random
import skimage.measure
import sys
import tempfile
import tensorflow as tf

# My libraries (alphabetical)
from data.utils import interpolate_zeros
from tools.tools import read_data
from utils import evenly_divisible_shape

FLAGS = None

max_input_size = 1200
border_trim = 100  # some lidar images are rotated, so that the edges are really weird. This arbitrary number is to crop the border, to try to avoid those weird edges
batch_size = 3
standard_input_size = 704  # set this to None to allow the use of full size images



def deepnn(x, x_shape, batch_size_holder):
    """deepnn builds the graph for a deep net for determining topography from image data.

  Args:
    x: input tensor with dimensions [batch_size, x_shape[0], x_shape[1], 3]
    x_shape: [width, height] of input image
    batch_size_holder: [batch_size]

  Returns:
    A tensor y of shape (output_size, output_size), with values equal to the topography of the input image
  """

    # First convolutional layer - maps one image to a bunch of feature maps.
    with tf.name_scope('conv1'):
        conv1_maps = 175
        w_conv1 = weight_variable([5, 5, 3, conv1_maps])
        b_conv1 = bias_variable([conv1_maps])
        h_conv1 = relu(conv2d(x, w_conv1) + b_conv1)

    # Pooling layer - downsamples.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer
    with tf.name_scope('conv2'):
        conv2_maps = 250
        w_conv2 = weight_variable([5, 5, conv1_maps, conv2_maps])
        b_conv2 = bias_variable([conv2_maps])
        h_conv2 = relu(conv2d(h_pool1, w_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Third convolutional layer
    with tf.name_scope('conv3'):
        conv3_maps = 250
        w_conv3 = weight_variable([4, 4, conv2_maps, conv3_maps])
        b_conv3 = bias_variable([conv3_maps])
        h_conv3 = relu(conv2d(h_pool2, w_conv3) + b_conv3)
#
    # Third pooling layer.
    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)

    # Fourth convolutional layer
    with tf.name_scope('conv4'):
        conv4_maps = 1
        w_conv4 = weight_variable([4, 4, conv3_maps, conv4_maps])
        b_conv4 = bias_variable([conv4_maps])
        h_conv4 = relu(conv2d(h_pool3, w_conv4) + b_conv4)

    # Fourth pooling layer.
    with tf.name_scope('pool4'):
        h_pool4 = max_pool_2x2(h_conv4)

    # First deconvolution layer. Upsample the last pooling layer and halve the number of feature maps
    deepnn.n_conv_layers = 4
    deconv_1_x = tf.cast(tf.ceil(tf.divide(x_shape[0], (pow(2, deepnn.n_conv_layers)))), tf.int32)
    deconv_1_y = tf.cast(tf.ceil(tf.divide(x_shape[1], (pow(2, deepnn.n_conv_layers)))), tf.int32)
    '''
    with tf.name_scope('deconv1'):
        stride1 = 2
        deconv1_maps = 50
        w_deconv1 = weight_variable([6, 6, deconv1_maps, conv4_maps])  # height, width, out channels, in channels
        out_shape = [batch_size_holder[0], deconv_1_x * stride1, deconv_1_y * stride1, deconv1_maps]
        h_deconv1 = relu(deconv2d(h_pool4, w_deconv1, out_shape, [1, stride1, stride1, 1]))

    # Second deconvolutional layer
    with tf.name_scope('deconv2'):
        stride2 = 2
        deconv2_maps = 25
        w_deconv2 = weight_variable([5, 5, deconv2_maps, deconv1_maps])
        out_shape = [batch_size_holder[0], deconv_1_x * stride1 * stride2, deconv_1_y * stride1 * stride2, deconv2_maps]
        h_deconv2 = relu(deconv2d(h_deconv1, w_deconv2, out_shape, [1, stride2, stride2, 1]))

    # Third deconvolutional layer
    with tf.name_scope('deconv3'):
        stride3 = 2
        w_deconv3 = weight_variable([4, 4, 1, deconv2_maps])
        out_shape = [batch_size_holder[0], deconv_1_x * stride1 * stride2 * stride3, deconv_1_y * stride1 * stride2 * stride3, 1]
        h_deconv3 = relu(deconv2d(h_deconv2, w_deconv3, out_shape, [1, stride3, stride3, 1]))

    final_size = (deconv_1_x * stride1 * stride2 * stride3, deconv_1_y * stride1 * stride2 * stride3)
    '''
    final_size = (deconv_1_x, deconv_1_y)
    print(final_size)
    with tf.name_scope('output'):
        # For use in the loss function, and in the inference script, rearrange the last layer to remove batch size and
        # the reference to the single channel
        # y_conv = tf.reshape(h_deconv3, [-1, deepnn.output_size, deepnn.output_size], name="final_op")
        y_conv = tf.reshape(h_pool4, [-1, final_size[0], final_size[1]], name="final_op")
    return y_conv

def relu(x):
    """ returns a relu. Meant to be easily modified to return a leaky relu """
    return tf.nn.relu(x)

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


def calc_newshape(shape):
    # Use the standard input size, if it isn't none, otherwise bring it down to a size evenly divisible by 2^(number of convolutional layers)
    # which makes it straightforward to predict the size of the output
    new_size = evenly_divisible_shape(shape, pow(2, deepnn.n_conv_layers))
    newx = standard_input_size if standard_input_size is not None else min(new_size[0], max_input_size)
    newy = standard_input_size if standard_input_size is not None else min(new_size[1], max_input_size)
    return (newx, newy)


def import_data_file(directory, type='completed'):
    topo_filename = [x for x in os.listdir(os.path.join(FLAGS.data_dir, type, directory)) if x[-5:] == '.data'][0]
    data = read_data(os.path.join(FLAGS.data_dir, type, directory, topo_filename))
    if any(dimension < ((border_trim * 2) * 1.5) for dimension in data.shape):
       del data
       return None
    data = data[border_trim:-border_trim, border_trim:-border_trim]
    data = np.flipud(data) # flip it around so that the data points match the pixel layout
    # crop the data as appropriate
    newx, newy = calc_newshape(data.shape)
    data = data[0:newx, 0:newy]
    reduced_data = skimage.measure.block_reduce(data, block_size=(16, 16), func=np.mean) # pool down
    interpolate_zeros(reduced_data)
    if reduced_data[0][0] == -999:
        print("Detected datafile with bad row in", directory)
        del data
        return None
    # It would be absurd to expect the network to learn absolute topography, which is what I believe this data is,
    # so we normalize the data by subtracting the mean of itself from itself. This should enable the net to learn
    # relative topography. It's also helpful that it's a fast operation
    reduced_data = reduced_data.astype(float)
    reduced_data -= reduced_data.mean()
    normalization_factor = 600000  # this number came from an analysis of the entire set. goal is normalization, i.e. get the data in a range from -1 to 1
    reduced_data /= normalization_factor
    reduced_data /= 2 # this gets the data set into a range of -0.5 to 0.5, roughly
    reduced_data += 1 # This should guarantee that all out data is >0, i.e. within the range of a relu unit
    reduced_data *= 1000 # maybe I have a vanishing gradient problem and maybe this will help?
    del data
    return reduced_data


def import_jp2_file(directory, type='completed'):
    jp2_filename = os.path.join(FLAGS.data_dir, type, directory, 'cropped.jp2')
    try:
        jp2_file = glymur.Jp2k(jp2_filename).read()
        if any(dimension < ((border_trim * 2) * 1.5) for dimension in jp2_file.shape[:2]):
           del jp2_file
           return None
        jp2_file = jp2_file[border_trim:-border_trim, border_trim:-border_trim, :]
        newx, newy = calc_newshape(jp2_file.shape)
        jp2_file = jp2_file[0:newx, 0:newy, :]
        jp2_file = jp2_file.astype(float)
        jp2_file -= np.mean(jp2_file, axis=0)
        jp2_file /= np.std(jp2_file, axis=0)
    except Exception as e:
        print(str(e))
        jp2_file = None
    return jp2_file


def enqueue(directories, q):
    sys.stderr = open(str(os.getpid()) + '.err', 'w')
    sys.stdout = open(str(os.getpid()) + '.out', 'w')
    # print("Queue size: ", q._maxsize)
    for d in directories:
        jp2_file = import_jp2_file(d)
        data_file = import_data_file(d)
        if data_file is not None and jp2_file is not None:
            try:
                q.put({'image':jp2_file, 'topography': data_file}, block=True)
            except:
                sys.stderr.flush()
                del jp2_file, data_file
        sys.stdout.flush()
        sys.stderr.flush()


def get_useful_directories(type='completed'):
    data_directories = os.listdir(FLAGS.data_dir + '/' + type)
    def shape_ok(d):
        if standard_input_size is None: return True
        with open(FLAGS.data_dir + '/' + type + '/' + d + '/cropped', 'r') as f:
            shape = [int(x) for x in f.readline().split(',')]
            return all(x > standard_input_size + 2* border_trim for x in shape[:2])
    data_directories = [x for x in data_directories if shape_ok(x)]
    return data_directories


def main(_):
    # Create the model
    x = tf.placeholder(tf.float32, [None, None, None, 3], name="input")
    x_shape = tf.placeholder(tf.int32, [2], name="shape")
    batch_size_holder = tf.placeholder(tf.int32, [1], name="batch_size")

    # Build the graph for the deep net
    y_conv = deepnn(x, x_shape, batch_size_holder)
    #print("Output size:", deepnn.output_size)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, None, None])

    with tf.name_scope('loss'):
        loss = tf.losses.mean_squared_error(labels=y_,
                                             predictions=y_conv, reduction=tf.losses.Reduction.SUM)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    with tf.name_scope('R2'):
        ssres = tf.reduce_sum(tf.squared_difference(y_, y_conv))
        sstot = tf.reduce_sum(tf.square(y_ - tf.reduce_mean(y_)))
        R2 = 1 - ssres/sstot


    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    # Get a list of all the directories that contain good data
    data_directories = get_useful_directories()

    # Break off some of these directories into a separate test set
    total_data = len(data_directories)
    # also want to ensure that the total data is evenly divisible by number of batches and processes
    num_processes = 2  # number of processes used to populate the queue
    def lcm(a, b):
        return int((a*b) / gcd(a, b))
    total_data -= total_data % lcm(batch_size, num_processes)
    print(total_data)
    training_directories = data_directories[0:total_data]
    training_directories.sort(key= lambda x: random())  # x is unused, this is for shuffling

    chunk_size = int(len(training_directories) / num_processes)
    # Set up a Queue for asynchronously loading the data
    training_data_queue = Manager().Queue(100)
    args = [[training_directories[i:i+chunk_size], training_data_queue] for i in range(0, len(training_directories), chunk_size)]
    processes = []
    for arg in args:
        p = Process(target=enqueue, args=arg)
        p.start()
        processes.append(p)

    # grab batch_size items out of the queue to have them for the periodic accuracy evaluation
    evaluation_batch = {'images': [], 'topographies': []}
    dirs = get_useful_directories('evaluation')
    for d in dirs:
        jp2_file = import_jp2_file(d, 'evaluation')
        data_file = import_data_file(d, 'evaluation')
        if jp2_file is not None and data_file is not None:
            evaluation_batch['images'].append(jp2_file)
            evaluation_batch['topographies'].append(data_file)
        if  len(evaluation_batch['images']) < batch_size:
            break
    print("evaluation size:", len(evaluation_batch['images']))

    # Set up the code to save the network and its parameters
    saver = tf.train.Saver()
    model_directory = "tnn_model"
    model_name = "tnn"
    model_path = os.path.join(model_directory, model_name)
    backup_directory = "backup"
    backup_path = os.path.join(backup_directory, model_name)  # save to two location, read from one

    # In attempting to deal with GPU issues, I will try to get tensorflow to only allocate gpu memory as needed, instead
    # of having it allocate all the GPU memory, which is what it does by default
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    accuracy_log = open('accuracy.log', 'a')

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


        def evaluate(epoch, batch_number):
            predictions = y_conv.eval(feed_dict={
                x: evaluation_batch['images'], x_shape: evaluation_batch['images'][0].shape[:2], batch_size_holder: [batch_size]})
            train_accuracy = loss.eval(feed_dict={
                y_conv: predictions, y_: evaluation_batch['topographies']})
            coefficient_of_determination = R2.eval(feed_dict={
                y_conv: predictions, y_: evaluation_batch['topographies']})
            stats_str = 'epoch %d, batch_number %d, evaluation loss: %.12g, eval accuracy: %.3g\n' % (epoch, batch_number, train_accuracy, coefficient_of_determination)
            print(stats_str)
            accuracy_log.write(stats_str)
            accuracy_log.flush()


        def save():
            saver.save(sess, model_path)
            saver.save(sess, backup_path)


        def get_training_batch():
            training_batch = {'images': [], 'topographies': []}
            for i in range(batch_size):
                try:
                    training_data = training_data_queue.get(block=True)
                except queue.Empty as qe:
                    print("Queue is empty")
                    print([p.is_alive() for p in processes])
                    continue
                except:
                    print("Failed to get data from queue")
                    continue
                training_batch['images'].append(training_data['image'])
                training_batch['topographies'].append(training_data['topography'])
            return training_batch if len(training_batch['images']) == batch_size else None


        for epoch in range(11000):
            print("Running epoch", epoch)
            # Run training through the entire data set. Reaching the end of the dataset will be indicated by the thread
            # stopping and the queue getting emptied
            batch_number = 0
            while (True in [p.is_alive() for p in processes]) or training_data_queue.empty() == False:
                print("Creating batch... (queue size is", training_data_queue.qsize(),")")
                training_batch = get_training_batch()
                if training_batch is None:
                    print("Couldn't get training batch")
                    continue
                print("Creating batch...done. Data point:",training_batch['topographies'][0][1][1])
                print("Creating batch...done. Image point:",training_batch['images'][0][1,1,:])

                print("Running training", epoch, "-", batch_number, "/", total_data / batch_size, "...")
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                print(training_batch['images'][0].shape)
                print(training_batch['topographies'][0].shape)
                sess.run([train_step], feed_dict={x: training_batch['images'], x_shape: training_batch['images'][0].shape[:2], y_: training_batch['topographies'], batch_size_holder: [batch_size]},
                         options=run_options, run_metadata=run_metadata)
                print("Running training", epoch, "-", batch_number, "/", total_data / batch_size, "...done")
                train_writer.add_run_metadata(run_metadata, 'step %d - %d' % (epoch, batch_number))
                sys.stdout.flush()
                if(batch_number % 40 == 0):
                    evaluate(epoch, batch_number)
                    save()

                # not sure if Python garbage collector is lagging behind or something, but this thing seems to take up
                # a lot more RAM than I think it should. I'll manually delete data here to try to avoid eating up all RAM
                del training_batch
                del run_options
                del run_metadata
                batch_number += 1
            for p in processes:
                p.join()
            processes.clear()
            # Now shuffle training directories and restart the thread
            data_directories = get_useful_directories()
            total_data = len(data_directories)
            total_data -= total_data % lcm(batch_size, num_processes)
            training_directories = data_directories[0:total_data]
            chunk_size = int(len(training_directories) / num_processes)
            training_directories.sort(key= lambda x: random())  # x is unused
            args = [[training_directories[i:i + chunk_size], training_data_queue] for i in range(0, len(training_directories), chunk_size)]
            for arg in args:
                p = Process(target=enqueue, args=arg)
                p.start()
                processes.append(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/home/docker/data',
                        help='Directory for input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
