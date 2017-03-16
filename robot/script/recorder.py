#!/usr/bin/env python
"""
Subscriber:
    1. This node subscribes to the topic /zed/rgb/image_rect_color to save the image using cv2
    2. This node subscribes to the topic /zed/depth/depth_registered to save the depth image using cv2
    3. This node subscribes to the topic /cmd_vel to get filename for each image
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Joy
import os
from threading import Lock
import time


class Recorder(object):
    def __init__(self):
        rospy.loginfo('[*]Start recorder.')
        self.record = False
        self.bridge = CvBridge()
        self.twist = None
        # create folders
        self.RGB_PATH = '../RGB_DATA/'
        self.DEPTH_PATH = '../DEPTH_DATA/'
        self.create_folders()
        self.twist_lock = Lock()
        self.start_btn_curr = False
        self.start_btn_prev = False
        self.start_btn_pressed = False
        # rgb image
        rospy.Subscriber('/zed/rgb/image_rect_color', Image, self.save_rgb)
        # depth
        rospy.Subscriber('/zed/depth/depth_registered', Image, self.save_depth)
        # cmd_vel
        rospy.Subscriber('/cmd_vel', Twist, self.get_twist)
        # joy
        rospy.Subscriber('/joy', Joy, self.get_status)
        rospy.init_node('recorder')
        # keeps the node alive
        rospy.spin()

    def create_folders(self):
        if not os.path.exists(self.RGB_PATH):
            print('[!]Creating rgb data folder.')
            os.mkdir(self.RGB_PATH)

        if not os.path.exists(self.DEPTH_PATH):
            print('[!]Creating depth data folder.')
            os.mkdir(self.DEPTH_PATH)

    def save_rgb(self, rgb):
        if self.record:
            try:
                image = self.bridge.imgmsg_to_cv2(rgb)
                if self.twist is not None:
                    timestamp = rgb.header.stamp
                    v = self.twist.linear.x
                    r = self.twist.angular.z
                    with self.twist_lock:
                        filename = os.path.join(self.RGB_PATH, '%s_%s_%s.png' % (timestamp, v, r))
                    image = cv2.resize(image, (256, 256))
                    # save image
                    cv2.imwrite(filename, image)
            except CvBridgeError as error:
                print(error)

    def save_depth(self, depth):
        if self.record:
            try:
                image = self.bridge.imgmsg_to_cv2(depth)

                if self.twist is not None:
                    timestamp = depth.header.stamp
                    v = self.twist.linear.x
                    r = self.twist.angular.z
                    with self.twist_lock:
                        filename = os.path.join(self.DEPTH_PATH, '%s-%s-%s.png' % (timestamp, v, r))
                    # save image
                    cv2.imwrite(filename, image)
            except CvBridgeError as error:
                print(error)

    def get_twist(self, twist):
        with self.twist_lock:
            self.twist = twist

    def get_status(self, joy_cmd):
        start = joy_cmd.buttons[3]
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
            self.record = not self.record
            if self.record:
                rospy.loginfo('[*]Start recording.')
            else:
                rospy.loginfo('[*]Stop recording.')
        # print(self.start_btn_press)

if __name__ == '__main__':
    try:
        recorder = Recorder()
    except rospy.ROSInterruptException:
        pass
