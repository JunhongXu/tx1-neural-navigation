#!/usr/bin/env python

"""
NeuralNet subscribes to /zed/rgb/image_rect_color and take the images to make predictions.
NeuralNet subscirbes to /joy to check if neural network command is on.
NeuralNet publishes to 'neural_cmd' topic which is being subscribed by commander to change the twist.
"""

import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist
from tf_model import NeuralCommander
import cv_bridge
import tensorflow as tf
import cv2
from controller import *
import sys
from ca_msgs.msg import Bumper


class NeuralNet(object):
    def __init__(self, train_iter):
        rospy.loginfo('[*]Loading Neural Network')
        self.train_iter = train_iter
        self.model = NeuralCommander()
        self.neural_net_on = False
        self.twist_cmd = rospy.Publisher('/neural_cmd', Twist, queue_size=5)
        self.safety_cmd = rospy.Publisher('/safety', Bool, queue_size=5)
        self.sess = tf.Session()
        self.model.restore(self.sess, self.train_iter)
        self.bridge = cv_bridge.CvBridge()
        self.controller = PS3()
        self.safe = True
        rospy.Subscriber('/bumper', Bumper, self.reset)
        rospy.Subscriber('/zed/rgb/image_rect_color', Image, callback=self.predict)
        rospy.Subscriber('/joy', Joy, callback=self.toggle_nn)
        rospy.init_node('neural_commander')
        rospy.spin()

    def reset(self, data):
        """stop neural network until it is out of obstacle"""
        if self.neural_net_on:
            pass

    def toggle_nn(self, data):
        self.controller.update(data)
        events = self.controller.btn_events
        if 'R2_pressed' in events:
            self.neural_net_on = not self.neural_net_on
            if self.neural_net_on:
                rospy.loginfo('[*]Start Neural Network.')
            else:
                rospy.loginfo('[*]Stop Neural Network.')

    def predict(self, data):
        if self.neural_net_on:
            twist = Twist()
            x = self.bridge.imgmsg_to_cv2(data)
            x = cv2.resize(x, (128, 128))
            primary_pi, safety_pi = self.model.predict(self.sess, x.reshape(1, 128, 128, 3))
            twist.linear.x = primary_pi[0][0]*0.5
            twist.angular.z = primary_pi[0][1]*4.25

            if safety_pi > 0.9:
                rospy.loginfo('[!]UNSAFE SITUATION DETECTED!')
                self.safe = False
            else:
                rospy.loginfo('[*]Angular velocity: %s' % twist.angular.z)
                rospy.loginfo('[*]Linear velocity: %s' % twist.linear.x)
                self.safe = True

            # publish safety situation
            safety = Bool()
            safety.data = self.safe
            self.safety_cmd.publish(safety)
            self.twist_cmd.publish(twist)


if __name__ == '__main__':
    try:
        train_iter = int(sys.argv[2]) - 1
        train_iter = max(0, train_iter)
        nn = NeuralNet(train_iter)
    except rospy.ROSInterruptException:
        pass
