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

class Recorder(object):
    def __init__(self):
        rospy.loginfo('[*]Start recorder.')
        self.bridge = CvBridge()

        # depth
        rospy.Subscriber('/zed/depth/depth_registered', Image, self.record_img)
        rospy.init_node('r')
        # keeps the node alive
        rospy.spin()

    def record_img(self, data):
        try:
            depth = self.bridge.imgmsg_to_cv2(data)
            depth_img = np.array(depth, dtype=np.float32)
            H, W = depth_img.shape
            print(H, W)
            # only record non-zero data
            timestamp = data.header.stamp
            filename = 'data/%s.png' % timestamp
            # save image
            depth_img = np.nan_to_num(depth_img)
            depth_img[depth_img > 20] = 0
            depth_img[depth_img < 0] = 0
            # print(depth_img)
            print('LEFT', np.mean(depth_img[:, W//4]))
            print('MIDDLE_1', np.mean(depth_img[:, W//4:W//2]))
            print('MIDDLE_2', np.mean(depth_img[:, W//2:3*W//4]))
            print('RIGHT', np.mean(depth_img[:, 3*W//4:W]))

            # cv2.imwrite(filename, depth)

        except CvBridgeError as error:
            print(error)


if __name__ == '__main__':
    try:
        recorder = Recorder()
    except rospy.ROSInterruptException:
        pass
