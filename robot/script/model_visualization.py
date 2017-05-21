#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import tensorflow as tf
import cv2
from tensorflow.contrib.layers import *
from model.tf_model import NeuralCommander
import numpy as np
import os


class Visualizer(object):
    """
    This class listens to the RGB data that sends from ZED Node and applies visualization
    techniques to see the most activated area from a given image
    """
    def __init__(self, iteration=1):
        self.bridge = CvBridge()
        self.sess = tf.Session()
        self.model = NeuralCommander()
        self.f0, self.f1, self.f2, self.f3, self.map = self.__build_deconv()
        self.sess.run(tf.global_variables_initializer())
        cwd = os.getcwd()
        cwd = os.path.join(cwd, '..')
        os.chdir(cwd)
        self.layers = self.model.layers[:7][::2]
        if iteration >= 0:
            self.model.restore(self.sess, iteration)

        rospy.Subscriber('/zed/rgb/image_rect_color', Image, callback=self.visualize)
        rospy.init_node('visualizer')
        rospy.spin()

    def __build_deconv(self):
        feature0 = tf.placeholder(dtype=tf.float32, shape=[None, 16, 16, 1])
        feature1 = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 1])
        feature2 = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        feature3 = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 1])
        w_init = tf.ones_initializer()
        b_init = tf.zeros_initializer()
        with tf.variable_scope('visualization'):
            deconv1 = conv2d_transpose(feature0, 1, kernel_size=3, stride=2, weights_initializer=w_init,
                                       biases_initializer=b_init, padding='SAME', scope='deconv1')
            combined_feat1 = tf.multiply(deconv1, feature1)
            deconv2 = conv2d_transpose(combined_feat1, 1, kernel_size=3, stride=2, weights_initializer=w_init,
                                       biases_initializer=b_init, padding='SAME', scope='deconv2')
            combined_feat2 = tf.multiply(deconv2, feature2)
            deconv3 = conv2d_transpose(combined_feat2, 1, kernel_size=3, stride=2, weights_initializer=w_init,
                                       biases_initializer=b_init, padding='SAME', scope='deconv3')
            combined_feat3 = tf.multiply(feature3, deconv3)

            # scale
            min_val = tf.reduce_min(combined_feat3)
            max_val = tf.reduce_max(combined_feat3)
            combined_feat3 = (combined_feat3 - min_val)/(max_val - min_val)
        return feature0, feature1, feature2, feature3, combined_feat3

    def overaly(self, feat, x):
        idx = np.where(np.squeeze(feat) >= 0.05)
        x[idx] = [0, 0, 255] * (1- x[idx])
        cv2.imshow('feature', cv2.resize(feat, (256, 256)))
        cv2.imshow('image', cv2.resize(x, (256, 256)))
        cv2.waitKey(10)

    def visualize(self, data):
        x = self.bridge.imgmsg_to_cv2(data)
        x = cv2.resize(x, (128, 128))
        features = self.sess.run(self.layers, feed_dict={self.model.x: x.reshape(1, 128, 128, 3)})
        f3, f2, f1, f0 =np.mean(features[0], axis=-1), np.mean(features[1], axis=-1), np.mean(features[2], axis=-1), \
                        np.mean(features[3], axis=-1)
        combined_feat = self.sess.run(self.map, feed_dict={
            self.f0: np.expand_dims(f0, axis=-1),
            self.f1: np.expand_dims(f1, axis=-1),
            self.f2: np.expand_dims(f2, axis=-1),
            self.f3: np.expand_dims(f3, axis=-1)
        })[0]

        self.overaly(combined_feat, x)


if __name__ == '__main__':
    try:
        viz = Visualizer(iteration=0)
    except rospy.ROSInterruptException:
        pass

