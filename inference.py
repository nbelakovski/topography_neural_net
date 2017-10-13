#!/usr/bin/python3

"""A deep  classifier using convolutional layers.
"""
import glymur
import tensorflow as tf
import os
import sys

def main(args):
  # Import data
  pass  # placeholder

  model_directory = "tnn_model"
  model_name = "tnn"
  meta_name = os.path.join(model_directory,model_name) + '.meta'

  with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    saver = tf.train.import_meta_graph(meta_name)
    saver.restore(sess, tf.train.latest_checkpoint(model_directory))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("input:0")
    op = graph.get_tensor_by_name("fc1/final_op:0")

    jp2_filename = 'data/15/cropped.jp2'
    jp2_file = glymur.Jp2k(jp2_filename)
    data = jp2_file[0:1008, 0:990, :]

    feed_dict1 = {x: [data]}

    print("Running session")
    out = sess.run(op, feed_dict=feed_dict1)
    print(out)


if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])
