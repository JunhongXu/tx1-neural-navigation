from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from tensorflow.contrib import layers
import os

"""
Copied from mode/tf_model.py
"""

class NeuralCommander(object):
    def __init__(self, inpt_size=(128, 128, 3)):
        self.x = tf.placeholder(shape=(None, ) + inpt_size, name='image', dtype=tf.float32)
        # two elements, first one is velocity, last one is rotation
        self.y = tf.placeholder(shape=(None, 2), name='twist', dtype=tf.float32)

        self.is_training = tf.placeholder(tf.bool, name='is_train')

        self.layers = []

        # build the model
        self.pi = self.build()

        self.params = tf.trainable_variables()
        # print all parameters and ops
        for p in self.params:
            print('{}: {}'.format(p.name, p.get_shape()))

        # loss
        self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(self.y, self.pi),1)),
                                   name='loss')

        with tf.name_scope('step'):
            self.global_setp = tf.Variable(initial_value=0, name='iteration', trainable=False)

        self.saver = tf.train.Saver(var_list=tf.trainable_variables())

    def predict(self, sess, x):
        return sess.run(self.pi, feed_dict={self.x: x, self.is_training: False})

    def save(self, sess):
        self.saver.save(sess, save_path='../../checkpoint/cnn-model')

    def restore(self, sess):
        saver = tf.train.import_meta_graph('../checkpoint/cnn-model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('../checkpoint'))
        for v in self.params:
            print(v.name)
            print(sess.run(v))

    def build(self):
        print('[*]Building...')
        with tf.name_scope('normalize'):
            image = tf.div(self.x, 255)

        with tf.variable_scope('conv1') as scope:
            conv1 = layers.conv2d(image, 64, 3, scope=scope)
            self.layers.append(conv1)
        with tf.variable_scope('pool1'):
            pool1 = layers.max_pool2d(conv1, kernel_size=2, padding='SAME')
            self.layers.append(pool1)

        with tf.variable_scope('conv2') as scope:
            conv2 = layers.conv2d(pool1, num_outputs=64, kernel_size=3, scope=scope)
            self.layers.append(conv2)

        with tf.variable_scope('pool2'):
            pool2 = layers.max_pool2d(conv2, kernel_size=2, padding='SAME')
            self.layers.append(pool2)

        with tf.variable_scope('conv3') as scope:
            conv3 = layers.conv2d(pool2, num_outputs=128, kernel_size=3, scope=scope)
            self.layers.append(conv3)

        with tf.variable_scope('pool3'):
            pool3 = layers.max_pool2d(conv3, kernel_size=2, padding='SAME')
            self.layers.append(pool3)

        with tf.variable_scope('conv4') as scope:
            conv4 = layers.conv2d(pool3, 128, 3, scope=scope)
            self.layers.append(conv4)

        with tf.variable_scope('pool4'):
            pool4 = layers.max_pool2d(conv4, 2, padding='SAME')
            self.layers.append(pool4)

        with tf.variable_scope('fc1') as scope:
            flattened = layers.flatten(pool4)
            fc1 = layers.fully_connected(flattened, 256, scope=scope)
            self.layers.append(fc1)

        with tf.variable_scope('dropout'):
            dropout = layers.dropout(fc1, is_training=self.is_training)
            self.layers.append(dropout)

        with tf.variable_scope('fc2') as scope:
            # tanh layer to scale y
            y = layers.fully_connected(dropout, 2, scope=scope, activation_fn=tf.nn.tanh)
            self.layers.append(y)
        print('[*]Building complete')
        return y

