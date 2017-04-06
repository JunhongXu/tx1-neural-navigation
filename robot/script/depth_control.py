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
        # depth
        rospy.Subscriber('/zed/depth/depth_registered', Image, self.update_depth)
        self.pub = rospy.Publisher('/depth_control', Twist, queue_size=5)
        self.move_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        # keeps the node alive
        rospy.spin()

    def reject_nan_inf(self, data):
        data = data[~np.isnan(data)]
        data = data[~np.isinf(data)]
        return data

    def update_depth(self, data):
        try:
            depth = self.bridge.imgmsg_to_cv2(data)
            depth_img = np.array(depth, dtype=np.float32)
            print(depth_img)
            H, W = depth_img.shape
            print(H, W)
            info = np.zeros(3)
            sum_data = 0
            for i in range(0, 3):
                data = depth_img[:, i*W//3:(i+1)*W//3]
                data = self.reject_nan_inf(data)
                sum_data += data.shape[0]
                data = self.reject_outliers(data)
                info[i] = np.mean(data)
            print(info)
            print(sum_data/(H*W))
            # processing depth
            # depth_img = np.nan_to_num(depth_img)
            # depth_img[depth_img > 5] = 0
            # depth_img[depth_img <= 0] = 0
            # depth_img = depth_img[:H//10]
            # info = np.zeros(2)
            # for i in range(0, 2):
            #     data = self.reject_outliers(depth_img[:, i*W//2:(i+1)*W//2])
            #     data = np.nan_to_num(data)
            #     info[i] = np.mean(data)
            # index = np.where(info<1)[0]
            # if np.mean(info) < 0.5:
            #     self.twist.linear.x = 0.5 - 0.5* np.max(info)
            # else:
            #     self.twist.linear.x = 0.5
            # if index.shape[0] != 0:
            #     minimal_dist = np.argmin(info)
            #     if minimal_dist == 0:
            #         self.twist.angular.z = 4.5 - 2*info[0]
            #         self.twist.angular.z = -self.twist.angular.z
            #     else:
            #         self.twist.angular.z = 4.5 - 2*info[1]
            # else:
            #     self.twist.angular.z = 0.0
            #
            # self.pub.publish(self.twist)
            # self.move_pub.publish(self.twist)
            # print(index)
            # print(self.twist)
            # print(info)
        except CvBridgeError as error:
            print(error)

    def reject_outliers(self, data):
        return data[abs(data - np.mean(data)) < 2* np.std(data)]


if __name__ == '__main__':
    try:
        recorder = DepthController()
    except rospy.ROSInterruptException:
        pass
