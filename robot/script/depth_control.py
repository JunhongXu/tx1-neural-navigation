#!/usr/bin/env python
"""
Subscriber:
    1. This node subscribes to the topic /zed/rgb/image_rect_color to save the image using cv2
    2. This node subscribes to the topic /zed/depth/depth_registered to save the depth image using cv2
    3. This node subscribes to the topic /cmd_vel to get filename for each image
    4. This node subscribes to the topic /zed/odom to save the 3D orientation relative to zed_initial_frame
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Bool
import os
from threading import Lock
from controller import PS3
import sys
from ca_msgs.msg import Bumper
import numpy as np


class DepthController(object):
    def __init__(self):
        rospy.loginfo('[*]Start recorder.')
        rospy.init_node('r')
        self.bridge = CvBridge()
        self.twist = Twist()
        self.division = 6
        self.pub = rospy.Publisher('/depth_control', Twist, queue_size=5)
        # depth
        rospy.Subscriber('/zed/depth/depth_registered', Image, self.update_depth)
        # keeps the node alive
        rospy.spin()

    def reject_nan_inf(self, data):
        data = data[~np.isnan(data)]
        data = data[~np.isinf(data)]
        return data

    def count(self, data, dist=1.5):
        return data[data<=dist].shape[0]/data.shape[0]

    def update_depth(self, data):
        try:
            depth = self.bridge.imgmsg_to_cv2(data)
            depth_img = np.array(depth, dtype=np.float32)
            # print(depth_img)
            H, W = depth_img.shape
            left_win = depth_img[:, :W//3]
            left_win = self.reject_outliers(self.reject_nan_inf(left_win))
            center_win = self.reject_outliers(self.reject_nan_inf(depth_img[:, W//3:2*W//3]))
            right_win = self.reject_outliers(self.reject_nan_inf(depth_img[:, 2*W//3:]))
            if self.count(center_win) >= 0.3:
                if np.sum(left_win) < np.sum(right_win):
                    self.twist.angular.z = -4.5*self.count(left_win)
                elif np.sum(left_win) >= np.sum(right_win):
                    self.twist.angular.z = 4.5*self.count(right_win)
            # for checking edge
            elif self.count(left_win) > 0.2 or self.count(right_win) >= 0.2:
                if np.sum(left_win) < np.sum(right_win):
                    self.twist.angular.z = -4.5*self.count(left_win)
                elif np.sum(left_win) >= np.sum(right_win):
                    self.twist.angular.z = 4.5*self.count(right_win)
            else:
                copyright()
                self.twist.angular.z = 0.0
            # info = np.zeros(self.division)
            # sum_data = 0
            # whole_mean = np.mean(self.reject_nan_inf(depth_img))
            # for i in range(0, self.division):
            #     data = depth_img[:, i*W//self.division:(i+1)*W//self.division]
            #     data = self.reject_nan_inf(data)
            #     data = self.reject_outliers(data)
            #     sum_data += data.shape[0]
            #     # data = np.sort(data, kind='mergesort')[:]
            #     info[i] = np.mean(data)
            # if np.mean(info[2:4]) < 1.0:
            #     self.twist.linear.x = 0.0
            #     print('[!]Stop')
            #
            # if np.mean(info[0:2])/np.mean(info[-2:-1]) < 0.6:
            #     self.twist.angular.z = 4.5 - 2 * self.sigmoid(info[0]/whole_mean)
            #     self.twist.angular.z = -self.twist.angular.z
            #     print('Turn right')
            # elif np.mean(info[-2:-1])/np.mean(info[0:2]) < 0.6:
            #     self.twist.angular.z = 4.5 - 2 * self.sigmoid(info[-1]/whole_mean)
            #     print('Turn left')
            # elif np.any(info[2:4]/whole_mean<0.8):
            #     # compare left and right
            #     if np.min(info[0:2]) > np.min(info[-2:-1]):
            #         self.twist.angular.z = 2.5
            #         print('[!!]Turn left')
            #     else:
            #         self.twist.angular.z = -2.5
            #         print('[!!]Turn right')
            # else:
            #     self.twist.angular.z = 0.0

            self.twist.linear.x = 0.5
            self.pub.publish(self.twist)
            # self.move_pub.publish(self.twist)
        except CvBridgeError as error:
            print(error)
        except ZeroDivisionError as error:
            print(error)

    def reject_outliers(self, data):
        return data[abs(data - np.mean(data)) < 2* np.std(data)]


    def sigmoid(self, data):
        return 1/ (1+np.exp(-data))


if __name__ == '__main__':
    try:
        recorder = DepthController()
    except rospy.ROSInterruptException:
        pass
