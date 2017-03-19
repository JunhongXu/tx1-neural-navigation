#!/usr/bin/env python

"""
NeuralNet subscribes to /zed/rgb/image_rect_color and take the images to make predictions.
NeuralNet subscirbes to /joy to check if neural network command is on.
NeuralNet publishes to 'neural_cmd' topic which is being subscribed by commander to change the twist.
"""

import rospy
from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist
from tf_model import NeuralCommander
import cv_bridge
import tensorflow as tf
import cv2
from controller import *


class NeuralNet(object):
    def __init__(self):
        rospy.loginfo('[*]Loading Neural Network')
        self.model = NeuralCommander()
        self.neural_net_on = False
        self.twist_cmd = rospy.Publisher('/neural_cmd', Twist)
        self.sess = tf.Session()
        self.model.restore(self.sess)
        self.bridge = cv_bridge.CvBridge()
        self.start_btn_prev = False
        self.start_btn_curr = False
        rospy.Subscriber('/zed/rgb/image_rect_color', Image, callback=self.predict)
        rospy.Subscriber('/joy', Joy, callback=self.toggle_nn)
        rospy.init_node('neural_commander')
        rospy.spin()

    def toggle_nn(self, data):
        start = data.buttons[9]
        self.start_btn_prev = self.start_btn_curr
        self.start_btn_curr = start == 1
        if self.start_btn_curr != self.start_btn_prev:
            if self.start_btn_curr:
                self.start_btn_pressed = True
            else:
                self.start_btn_pressed = False
        else:
            self.start_btn_pressed = False

        if self.start_btn_pressed:
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
            cmd = self.model.predict(self.sess, x.reshape(1, 128, 128, 3))
            rospy.loginfo('Predicted steer command: linear: %s, angular: %s' %(cmd[0][0]*0.5, cmd[0][1]*4.25))
            twist.linear.x = cmd[0][0]*0.5
            twist.angular.z = cmd[0][1]*4.25
            # publish the command
            self.twist_cmd.publish(twist)

if __name__ == '__main__':
    try:
        nn = NeuralNet()
    except rospy.ROSInterruptException:
        pass
