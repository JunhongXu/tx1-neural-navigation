from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from tensorflow.contrib import layers


class NeuralCommander(object):
    def __init__(self, inpt_size=(128, 128, 3)):
        self.x = tf.placeholder(shape=(None, ) + inpt_size, name='image', dtype=tf.float32)
        self.safety_inpt = tf.placeholder(shape=(None, 256), dtype=tf.float32, name='safety_inpt')
        # two elements, first one is velocity, last one is rotation
        self.y = tf.placeholder(shape=(None, 2), name='twist', dtype=tf.float32)
        self.safety_y = tf.placeholder(shape=(None, 1), name='safety_y', dtype=tf.float32)

        self.is_training = tf.placeholder(tf.bool, name='is_train')

        self.layers = []

        # build the model
        self.pi = self.build()

        # build the safety policy
        self.safety_pi = self.build_safety()

        self.safety_logit = tf.nn.sigmoid(self.safety_pi)

        self.params = tf.trainable_variables()
        # print all parameters and ops
        for p in self.params:
            print('{}: {}'.format(p.name, p.get_shape()))

        # primary loss
        with tf.variable_scope('primary_loss'):
            self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(self.y, self.pi), 1)), name='loss')

        # safety loss
        self.safety_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.safety_y,
                                                                                  logits=self.safety_pi,
                                                                                  name='safety_loss'))

        with tf.name_scope('step'):
            self.global_pi_setp = tf.Variable(initial_value=0, name='iteration', trainable=False)
            self.global_safety_step = tf.Variable(initial_value=0, name='safety_iter', trainable=False)

        self.saver = tf.train.Saver(var_list=tf.trainable_variables())

    def predict(self, sess, x):
        # predict the primary policy and feature vector
        primary_pi, feature = sess.run([self.pi, self.layers[-3]], feed_dict={self.x: x, self.is_training: False})

        # predict the safety policy
        safety = sess.run(self.safety_logit, feed_dict={self.safety_inpt: feature, self.is_training: False})
        return primary_pi, safety

    def save(self, sess, num_iter):
        self.saver.save(sess, save_path='../checkpoint/%s/cnn-model' % num_iter)
        print('[*]Completely saved model.')

    def restore(self, sess, num_iter):
        ckpt = tf.train.get_checkpoint_state('../checkpoint/%s/' % num_iter)
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('...no checkpoint found...')

    def build_safety(self):
        with tf.variable_scope('safety_policy'):
            with tf.variable_scope('fc1') as scope:
                fc1 = layers.fully_connected(self.safety_inpt, 200, scope=scope)

            with tf.variable_scope('dropoout'):
                dropout = layers.dropout(fc1, keep_prob=0.5, is_training=self.is_training)

            with tf.variable_scope('fc2') as scope:
                fc2 = layers.fully_connected(dropout, 1, scope=scope, activation_fn=None)

        return fc2

    def build(self):
        print('[*]Building...')
        with tf.variable_scope('cnn'):
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

