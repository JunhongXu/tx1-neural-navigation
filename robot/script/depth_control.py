#!/usr/bin/env python
"""
Subscriber:
    1. This node subscribes to the topic /zed/rgb/image_rect_color to save the image using cv2
    2. This node subscribes to the topic /zed/depth/depth_registered to save the depth image using cv2
    3. This node subscribes to the topic /cmd_vel to get filename for each image
    4. This node subscribes to the topic /zed/odom to save the 3D orientation relative to zed_initial_frame
    5. This node subscribes to the topic /distance_left and /distance_right to detect possible collisions
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Float32
import numpy as np


class DepthController(object):
    def __init__(self):
        rospy.loginfo('[*]Start recorder.')
        rospy.init_node('r')
        self.bridge = CvBridge()
        self.twist = Twist()
        self.division = 6
        self.pub = rospy.Publisher('/depth_control', Twist, queue_size=5)
        self.left_dist = Float32()
        self.right_dist = Float32()
        self.is_close = False
        self.safety_distance = 0.2
        # depth
        rospy.Subscriber('/zed/depth/depth_registered', Image, self.update_depth)
        rospy.Subscriber('/distance_left', Float32, self.update_left_distance)
        rospy.Subscriber('/distance_right', Float32, self.update_right_distance)
        # keeps the node alive
        rospy.spin()

    def update_left_distance(self, data):
        self.left_dist = data.data

    def update_right_distance(self, data):
        self.right_dist = data.data


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
            self.twist.linear.x = 0.5
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
            # for checking ultrosonic distance
            elif self.left_dist <=self.safety_distance or self.right_dist <=self.safety_distance:
                # this actually is right, have wrong setup in the hardware
                if self.left_dist <= self.safety_distance:
                    self.twist.angular.z = 4.5
                elif self.right_dist <= self.safety_distance:
                    self.twist.angular.z = -4.5
                elif self.right_dist <= self.safety_distance and self.left_dist <=self.safety_distance:
                    self.twist.linear.x = -0.2
            else:
                self.twist.angular.z = 0.0
            self.pub.publish(self.twist)
        except CvBridgeError as error:
            print(error)
        except ZeroDivisionError as error:
            print(error)

    def reject_outliers(self, data):
        return data[abs(data - np.mean(data)) < 2* np.std(data)]


if __name__ == '__main__':
    try:
        recorder = DepthController()
    except rospy.ROSInterruptException:
        pass
