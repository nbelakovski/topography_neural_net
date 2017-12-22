#!/usr/bin/python3

# Attribution: This code started from the Tensorflow deep MNIST convulational
# example, and was expanded upon from there.

"""A deep regressor using convolutional layers."""

# Python libraries (alphabetical)
import argparse
from collections import defaultdict
from glob import glob
import glymur
from math import ceil, gcd
from multiprocessing import Queue, Process, Manager
from FIFORedisQueue import Queue as MyQueue
import numpy as np
import os
import queue
from random import random
import skimage.measure
import sys
import tempfile
import tensorflow as tf

# My libraries (alphabetical)
from data.utils import interpolate_zeros_2
from tools.tools import read_data
from utils import evenly_divisible_shape

FLAGS = None
max_input_size = 1200
border_trim = 100  # some lidar images are rotated, so that the edges are really weird. This arbitrary number is to crop the border, to try to avoid those weird edges
batch_size = 3
standard_input_size = 720  # set this to None to allow the use of full size images


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

    final_size = (deconv_1_x, deconv_1_y)
    print(final_size)
    with tf.name_scope('output'):
        # For use in the loss function, and in the inference script, rearrange the last layer to remove batch size and
        # the reference to the single channel
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


"""
============================================================================================================
============================================================================================================
============================================================================================================
START FUNCTION FOR LOADING DATA INTO MEMORY
============================================================================================================
============================================================================================================
============================================================================================================
"""


def load_data_file(data_info, type='training'):
    directory = data_info['directory']
    topo_filename = glob(os.path.join(FLAGS.data_dir, type, directory, '*data'))[0]
    data = read_data(os.path.join(FLAGS.data_dir, type, directory, topo_filename))
    return data

def crop_data_file(data, data_info):
    # crop the data as appropriate
    newx, newy = calc_newshape(data.shape)
    if data_info['translation'] == "UL":
        data = data[0:newx, 0:newy]
    elif data_info['translation'] == "UR":
        data = data[0:newx, -newy:]
    elif data_info['translation'] == "BL":
        data = data[-newx:, 0:newy]
    elif data_info['translation'] == "BR":
        data = data[-newx:, -newy:]
    return data

def normalize_data_file(data):
    # It would be absurd to expect the network to learn absolute topography, which is what I believe this data is,
    # so we normalize the data by subtracting the mean of itself from itself. This should enable the net to learn
    # relative topography. It's also helpful that it's a fast operation
    data = data.astype(float)
    data -= data.mean()
    normalization_factor = 600000  # this number came from an analysis of the entire set. goal is normalization, i.e. get the data in a range from -1 to 1
    data /= normalization_factor
    data /= 2 # this gets the data set into a range of -0.5 to 0.5, roughly
    data += 1 # This should guarantee that all out data is >0, i.e. within the range of a relu unit
    data *= 1000 # maybe I have a vanishing gradient problem and maybe this will help?
    return data

def import_data_file(data_info, type='training'):
    data = load_data_file(data_info, type)
    data = data[border_trim:-border_trim, border_trim:-border_trim]
    data = np.flipud(data) # flip it around so that the data points match the pixel layout
    data = crop_data_file(data, data_info)
    data = np.rot90(data, data_info['rotation']/90)
    interpolate_zeros_2(data)
    data = skimage.measure.block_reduce(data, block_size=(16, 16), func=np.mean) # pool down
    data = normalize_data_file(data)
    return data


"""
============================================================================================================
LOADING JP2 FILE FUNCTIONS
============================================================================================================
"""


def load_jp2_file(image_info, type='training'):
    directory = image_info['directory']
    jp2_filename = os.path.join(FLAGS.data_dir, type, directory, 'cropped.jp2')
    try:
        return glymur.Jp2k(jp2_filename).read()
    except Exception as e:
        print(str(e))
        return None

def crop_jp2_file(jp2_file, image_info):
    newx, newy = calc_newshape(jp2_file.shape)
    # Translate here:
    if image_info['translation'] == "UL":
        jp2_file = jp2_file[0:newx, 0:newy, :]
    elif image_info['translation'] == "UR":
        jp2_file = jp2_file[0:newx, -newy:, :]
    elif image_info['translation'] == "BL":
        jp2_file = jp2_file[-newx:, 0:newy, :]
    elif image_info['translation'] == "BR":
        jp2_file = jp2_file[-newx:, -newy:, :]
    return jp2_file

def normalize_jp2_file(jp2_file):
    jp2_file = jp2_file.astype(float)
    jp2_file -= np.mean(jp2_file, axis=0)
    jp2_file /= np.std(jp2_file, axis=0)
    return jp2_file

def import_jp2_file(image_info, type='training'):
    jp2_file = load_jp2_file(image_info, type)
    if jp2_file is not None:
        jp2_file = jp2_file[border_trim:-border_trim, border_trim:-border_trim, :]
        jp2_file = crop_jp2_file(jp2_file, image_info)
        jp2_file = np.rot90(jp2_file, image_info['rotation']/90)
        jp2_file = normalize_jp2_file(jp2_file)
    else:
        return None
    return jp2_file


"""
============================================================================================================
============================================================================================================
============================================================================================================
DONE FUNCTIONS FOR LOADING DATA INTO MEMORY
============================================================================================================
============================================================================================================
============================================================================================================
"""


def enqueue(directories, q, training_or_evaluation):
    pid = os.getpid()
    total = 0
    for d in directories:
        jp2_file = import_jp2_file(d, training_or_evaluation)
        data_file = import_data_file(d, training_or_evaluation)
        if data_file is not None and jp2_file is not None:
            print(pid, "Putting...", q)
            try:
                q.put({'image':jp2_file, 'topography': data_file}, timeout=300)
                total += 1
            except Exception as e:
                print(pid, "failed to put", str(e))
            print(pid, total)
        sys.stdout.flush()
        sys.stderr.flush()
    print(pid, "Done")


def get_useful_directories(training_or_evaluation):
    data_directories = os.listdir(FLAGS.data_dir + '/' + training_or_evaluation)
    def shape_ok(d):
        if standard_input_size is None: return True
        with open(FLAGS.data_dir + '/' + training_or_evaluation + '/' + d + '/cropped_size.txt', 'r') as f:
            shape = [int(x) for x in f.readline().split(',')]
            return all(x > (standard_input_size + 2* border_trim) for x in shape[:2])
    data_directories = [x for x in data_directories if shape_ok(x)]
    return data_directories


def lcm(a, b):
        return int((a*b) / gcd(a, b))


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

    num_processes = 3  # number of processes used to populate the queue
    # Set up various parameters for the training set and evaluation set
    training_directories = get_useful_directories('training')
    # Inflate the directories by a factor of 4 by adding a rotation to each one
    def add_data_augmentation_information(d):
        rval = []
        # rotations and translations:
        for rotation in [0, 90, 180, 270]:
            for translation in ["UL", "UR", "BL", "BR"]:
                rval.append({'directory': d, 'rotation': rotation, 'translation': translation})
        return rval
    def augment_data(directories):
        new_directories = []
        for d in directories:
            new_directories.extend(add_data_augmentation_information(d))
        return new_directories

    print("total data:", len(training_directories))
    training_directories = augment_data(training_directories)
    print("augmented data:", len(training_directories))
    total_training_data = len(training_directories) - len(training_directories) % lcm(batch_size, num_processes)
    training_directories = training_directories[0:total_training_data]
    training_chunk_size = int(len(training_directories) / num_processes)

    evaluation_dirs = get_useful_directories('evaluation')
    evaluation_dirs = [{'directory': d, 'rotation': 0, 'translation': 'UL'} for d in evaluation_dirs]
    total_eval_data = len(evaluation_dirs) - len(evaluation_dirs) % lcm(batch_size, num_processes)
    evaluation_dirs = evaluation_dirs[0:total_eval_data]
    eval_chunk_size = int(len(evaluation_dirs) / num_processes)

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


        def get_evaluation_batch():
            evaluation_batch = {'images': [], 'topographies': []}
            for i in range(batch_size):
                d = evaluation_dirs[int(random() * len(evaluation_dirs))]
                jp2_file = import_jp2_file(d, 'evaluation')
                data_file = import_data_file(d, 'evaluation')
                if jp2_file is not None and data_file is not None:
                    evaluation_batch['images'].append(jp2_file)
                    evaluation_batch['topographies'].append(data_file)
            return evaluation_batch



        def evaluate(epoch, batch_number, batch):
            predictions = y_conv.eval(             feed_dict={x: batch['images'], x_shape: batch['images'][0].shape[:2], batch_size_holder: [batch_size]})
            train_accuracy = loss.eval(            feed_dict={y_conv: predictions, y_: batch['topographies']})
            coefficient_of_determination = R2.eval(feed_dict={y_conv: predictions, y_: batch['topographies']})
            stats_str = 'epoch %d, batch_number %d, evaluation loss: %.12g, eval accuracy: %.3g\n' % (epoch, batch_number, train_accuracy, coefficient_of_determination)
            print(stats_str)
            accuracy_log.write(stats_str)
            accuracy_log.flush()
            return train_accuracy, coefficient_of_determination


        def save():
            saver.save(sess, model_path)
            saver.save(sess, backup_path)

        total_received = defaultdict(lambda: 0)
        def get_batch(queue):
            batch = {'images': [], 'topographies': []}
            for i in range(batch_size):
                try:
                    data = queue.get(block=True, timeout=300)
                    total_received[str(queue)] += 1
                    batch['images'].append(data['image'])
                    batch['topographies'].append(data['topography'])
                except Exception as e:
                    print("Failed to get", str(e))
            return batch if len(batch['images']) == batch_size else None



        for epoch in range(10):
            print("Running epoch", epoch)
            # Shuffle training directories and kick off processes to populate queue
            training_directories.sort(key= lambda x: random())  # x is unused
            training_data_queue = MyQueue(300, name='training'+str(epoch))
            print("Training data queue:", training_data_queue)
            args = [[training_directories[i:i + training_chunk_size], training_data_queue, 'training'] for i in range(0, len(training_directories), training_chunk_size)]
            processes = []
            for arg in args:
                p = Process(target=enqueue, args=arg)
                p.start()
                processes.append(p)

            # Run training through the entire data set. Reaching the end of the dataset will be indicated by the processes
            # stopping and the queue getting emptied
            batch_number = 0
            while (True in [p.is_alive() for p in processes]) or training_data_queue.empty() == False:
                print("Creating batch... (queue size is", training_data_queue.qsize(),")")
                training_batch = get_batch(training_data_queue)
                if training_batch is None:
                    print("Couldn't get training batch")
                    continue
                print("Creating batch...done. Data point:", training_batch['topographies'][0][1][1])
                print("Creating batch...done. Image point:", training_batch['images'][0][1,1,:])

                print("Running training", epoch, "-", batch_number, "/", total_training_data / batch_size, "...")
                print(training_batch['images'][0].shape)
                print(training_batch['topographies'][0].shape)
                sess.run([train_step], feed_dict={x: training_batch['images'],
                                                  x_shape: training_batch['images'][0].shape[:2],
                                                  y_: training_batch['topographies'],
                                                  batch_size_holder: [batch_size]})
                print("Running training", epoch, "-", batch_number, "/", total_training_data / batch_size, "...done")
                sys.stdout.flush()
                if(batch_number % 200 == 0):
                    evaluate(epoch, batch_number, get_evaluation_batch())
                    save()

                batch_number += 1
            print("Epoch", epoch, "completed, evaluating...")
            for p in processes:
                p.join()
            processes.clear()
            # At the end of an epoch, evaluate over the entire evaluation set
            evaluation_data_queue = MyQueue(300, name='evaluation'+str(epoch))
            print("Evaluation data queue:", evaluation_data_queue)
            args = [[evaluation_dirs[i:i + eval_chunk_size], evaluation_data_queue, 'evaluation'] for i in range(0, len(evaluation_dirs), eval_chunk_size)]
            for arg in args:
                p = Process(target=enqueue, args=arg)
                p.start()
                processes.append(p)
            batch_number = 0
            accuracies = []
            coeffs = []
            while (True in [p.is_alive() for p in processes]) or evaluation_data_queue.empty() == False:
                accuracy, coeff = evaluate(epoch, batch_number, get_batch(evaluation_data_queue))
                accuracies.append(accuracy)
                coeffs.append(coeff)
                batch_number += 1
            print("Accuracy after epoch", epoch, ":", (sum(accuracies)/len(accuracies)))
            print("R2 after epoch", epoch, ":", (sum(coeffs)/len(coeffs)))

            # Clean up
            for p in processes:
                p.join()
            processes.clear()

            # Should add code to save here, so that I can keep track of the network via epochs, and just use the eval save for some inference



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/home/docker/data',
                        help='Directory for input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
