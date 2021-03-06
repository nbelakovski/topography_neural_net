#!/usr/bin/python3

"""A deep  regressor using convolutional layers.
"""
import glymur
import numpy as np
import pickle
import tensorflow as tf
import os
import sys
import utils

def main(args):
  # Import data

  model_directory = "inference_save"
  model_name = "tnn"
  meta_name = os.path.join(model_directory,model_name) + '.meta'
  device = None if len(sys.argv) == 3 and sys.argv[2] == "--gpu" else '/cpu:0'
  with tf.device(device):
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
      sess.run(tf.global_variables_initializer())
      saver = tf.train.import_meta_graph(meta_name)
      saver.restore(sess, tf.train.latest_checkpoint(model_directory))
      graph = tf.get_default_graph()
      x = graph.get_tensor_by_name("input:0")
      x_shape = graph.get_tensor_by_name("shape:0")
      batch_size= graph.get_tensor_by_name("batch_size:0")
      op = graph.get_tensor_by_name("output/final_op:0")

      jp2_filename = sys.argv[1]
      jp2_file = glymur.Jp2k(jp2_filename).read()
      # crop it down to a shape that's evenly divisible by the total amount of pooling it will go through
      # this helps avoid weird padding issues. I could change the padding in the net, but I'd rather set
      # up the input so that it's a non-issue
      new_shape = utils.evenly_divisible_shape(jp2_file.shape, 16)  # TODO: don't hardcode 16 :(
      jp2_file = jp2_file[0:new_shape[0], 0:new_shape[1], :]

      feed_dict1 = {x: [jp2_file], x_shape: jp2_file.shape[:2], batch_size: [1]}

      print("Running session")
      out = sess.run(op, feed_dict=feed_dict1)
      print(out[0])
      print("Nonzero count:", np.count_nonzero(out[0]),"/",out[0].size)
      pickle.dump(out[0], open('test.pickle', 'wb'))


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
