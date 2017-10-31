#!/usr/bin/python3

"""A deep  regressor using convolutional layers.
"""
import glymur
import pickle
import tensorflow as tf
import os
import sys

def main(args):
  # Import data

  model_directory = "tnn_model"
  model_name = "tnn"
  meta_name = os.path.join(model_directory,model_name) + '.meta'

  with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    saver = tf.train.import_meta_graph(meta_name)
    saver.restore(sess, tf.train.latest_checkpoint(model_directory))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("input:0")
    op = graph.get_tensor_by_name("output/final_op:0")

    jp2_filename = 'sample_data/01824c/cropped.jp2'
    jp2_file = glymur.Jp2k(jp2_filename)
    data = jp2_file[0:800, 0:800, :]

    batch_size = 5
    input_array = [data for x in range(batch_size)]
    feed_dict1 = {x: input_array}

    print("Running session")
    out = sess.run(op, feed_dict=feed_dict1)
    print(out[0])
    pickle.dump(out[0], open('test.pickle', 'wb'))


if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])
