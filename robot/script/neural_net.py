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


class NeuralNet(object):
    def __init__(self, train_iter):
        rospy.loginfo('[*]Loading Neural Network')
        self.train_iter = train_iter
        self.model = NeuralCommander()
        self.neural_net_on = False
        self.twist_cmd = rospy.Publisher('/neural_cmd', Twist)
        self.safety_cmd = rospy.Publisher('/safety', Bool)
        self.human_control = rospy.Publisher('/human', Bool)
        self.human_twist = rospy.Publisher('/human_vel', Twist, queue_size=1)
        self.human_twist_cmd = Twist()
        self.sess = tf.Session()
        self.model.restore(self.sess, self.train_iter)
        self.bridge = cv_bridge.CvBridge()
        self.controller = PS3()
        self.safe = True
        self.human_cmd = False
        rospy.Subscriber('/zed/rgb/image_rect_color', Image, callback=self.predict)
        rospy.Subscriber('/joy', Joy, callback=self.toggle_nn)
        rospy.init_node('neural_commander')
        rospy.spin()

    def toggle_nn(self, data):
        self.controller.update(data)
        events = self.controller.btn_events
        if 'R2_pressed' in events:
            self.neural_net_on = not self.neural_net_on
            if self.neural_net_on:
                rospy.loginfo('[*]Start Neural Network.')
            else:
                rospy.loginfo('[*]Stop Neural Network.')

        if 'Circle_pressed' in events and not self.safe:
            self.human_cmd = True

            rospy.loginfo('[*]Human labelling...')
        if self.human_cmd:
            vel = self.controller.left_stick*0.5
            angular = self.controller.right_stick*4.5
            self.human_twist_cmd.linear.x = vel
            self.human_twist_cmd.angular.z = angular

    def predict(self, data):
        if self.neural_net_on:
            twist = Twist()
            x = self.bridge.imgmsg_to_cv2(data)
            x = cv2.resize(x, (128, 128))
            primary_pi, safety_pi = self.model.predict(self.sess, x.reshape(1, 128, 128, 3))
            twist.linear.x = primary_pi[0][0]*0.5
            twist.angular.z = primary_pi[0][1]*4.25

            if safety_pi > 0.8:
                if not self.human_cmd:
                    rospy.loginfo('[!]UNSAFE SITUATION DETECTED! Waiting for human instruction.')
                self.safe = False
            else:
                self.safe = True

            safety = Bool()
            safety.data = self.safe
            self.safety_cmd.publish(safety)

            if self.safe:
                # publish the command
                self.twist_cmd.publish(twist)
                self.human_cmd = False
                rospy.loginfo('[*]Steering command: linear: %s, angular: %s' %(primary_pi[0][0]*0.5, primary_pi[0][1]*4.25))
                rospy.loginfo('[*]Safety value: %s' % safety_pi)
            else:
                if self.human_cmd:
                    rospy.loginfo('[*]Steering command: linear: %s, angular: %s' %(primary_pi[0][0]*0.5, primary_pi[0][1]*4.25))
                    rospy.loginfo('[*]Safety value: %s' % safety_pi)
                    self.twist_cmd.publish(twist)
                    self.human_twist.publish(self.human_twist_cmd)
            self.human_control.publish(Bool(self.human_cmd))


if __name__ == '__main__':
    try:
        train_iter = int(sys.argv[2]) - 1
        nn = NeuralNet(train_iter)
    except rospy.ROSInterruptException:
        pass
