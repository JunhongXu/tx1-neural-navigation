#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import tensorflow as tf
import cv2
from tensorflow.contrib.layers import *
from model.tf_model import NeuralCommander
import numpy as np
import os
import time

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
        self.iteration = iteration
        cwd = os.getcwd()
        cwd = os.path.join(cwd, '..')
        os.chdir(cwd)
        self.layers = self.model.layers[:7][::2]
        if iteration >= 0:
            self.model.restore(self.sess, iteration)
        self.safety_value = Float32()
        self.twist = Twist()
        self.depth_control = Twist()
        rospy.Subscriber('/zed/rgb/image_rect_color', Image, callback=self.visualize)
        rospy.Subscriber('/depth_control', Twist, callback=self.depth)
        rospy.Subscriber('/neural_cmd', Twist, callback=self.get_twist)
        rospy.Subscriber('/safety_value', Float32, callback=self.get_safety)
        rospy.init_node('visualizer')
        rospy.spin()

    def depth(self, data):
        self.depth_control = data

    def get_twist(self, data):
        self.twist = data

    def get_safety(self, data):
        self.safety_value = data

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
        height = self.twist.linear.x/0.5
        width = self.twist.angular.z/4.5
        depth_height = -self.depth_control.linear.x/0.5
        depth_width = -self.depth_control.angular.z/4.5
        idx = np.where(np.squeeze(feat) >= 0.05)
        x[idx] = [0, 0, 255] * (1 - x[idx])
        safety = self.safety_value.data
        # print(safety)
        x = cv2.resize(x, (256, 256))
        x = cv2.copyMakeBorder(x, 15, 0, 150, 150, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.rectangle(x, (0, 0), (int((256+300)*safety), 15), (0, int((1 - safety)*255), int(safety * 255)), thickness=-1)
        cv2.rectangle(x, (60, int(-height*75) + 136), (90, 137), (255, 0, 0),  thickness=-1)
        if width < 0.0:
            cv2.rectangle(x, (int(-width*60)+90, 106), (90, 137), (0, 0, 255), thickness=-1)
        elif width > 0.0:
            cv2.rectangle(x, (int(-width*60)+60, 106), (60, 137), (0, 0, 255), thickness=-1)
        cv2.rectangle(x, (406+60, int(depth_height *75) + 136), (406+90, 137), (255, 0, 0),  thickness=-1)
        if depth_width < 0.0:
            cv2.rectangle(x, (int(depth_width*60)+60+406, 106), (60+406, 137), (0, 0, 255), thickness=-1)
        elif depth_width > 0.0:
            cv2.rectangle(x, (int(depth_width*60)+90+406, 106), (90+406, 137), (0, 0, 255), thickness=-1)
        cv2.putText(x, 'Neural Network Policy', (5, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 255, 255))
        cv2.putText(x, 'Sensor Policy', (436, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 255, 255))
        cv2.imwrite('/media/jxu7/BACK-UP/Data/neural-navigation/video_material/%s/%s.jpg' % (self.iteration, time.time()), x)
        cv2.imshow('feature', cv2.resize(feat, (256, 256)))
        cv2.imshow('image', x)

        cv2.waitKey(1)

    def visualize(self, data):
        x = self.bridge.imgmsg_to_cv2(data)
        x = cv2.resize(x, (128, 128))
        features = self.sess.run(self.layers, feed_dict={self.model.x: x.reshape(1, 128, 128, 3)})
        f3, f2, f1, f0 =np.mean(features[0], axis=-1), np.mean(features[1], axis=-1), np.mean(features[2], axis=-1), \
                        np.mean(features[3], axis=-1)
        combined_feat = self.sess.run(self.map,  feed_dict={
            self.f0: np.expand_dims(f0, axis=-1),
            self.f1: np.expand_dims(f1, axis=-1),
            self.f2: np.expand_dims(f2, axis=-1),
            self.f3: np.expand_dims(f3, axis=-1)
        })[0]

        self.overaly(combined_feat, x)


if __name__ == '__main__':
    try:
        viz = Visualizer(iteration=1)
    except rospy.ROSInterruptException:
        pass

