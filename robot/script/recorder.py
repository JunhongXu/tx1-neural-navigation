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


class Recorder(object):
    def __init__(self, train_iter):
        rospy.loginfo('[*]Start recorder.')
        self.record = False
        self.bridge = CvBridge()
        # create folders
        self.SAFETY_RGB_PATH = '../safety/RGB_DATA/%s' % train_iter
        self.SAFETY_DEPTH_PATH = '../safety/DEPTH_DATA/%s' % train_iter
        self.DEPTH_PATH = '../primary/DEPTH_DATA/%s' % train_iter
        self.RGB_PATH = '../primary/RGB_DATA/%s' % train_iter
        self.create_folders()
        self.twist_lock = Lock()
        self.controller = PS3()
        self.primary_record = False
        self.safety_record = False
        self.safe = True
        self.twist = Twist()

        rospy.Subscriber('/zed/rgb/image_rect_color', Image, self.save_rgb)
        # depth
        rospy.Subscriber('/zed/depth/depth_registered', Image, self.save_depth)
        # cmd_vel
        rospy.Subscriber('/cmd_vel', Twist, self.get_twist)
        # odom
        rospy.Subscriber('/odom', Odometry, self.save_odom)
        # joy
        rospy.Subscriber('/joy', Joy, self.get_status)
        # safety
        rospy.Subscriber('/safety', Bool, self.update_safety)
        # bumper msg
        rospy.Subscriber('/bumper', Bumper, self.reset)

        rospy.init_node('recorder')
        # keeps the node alive
        rospy.spin()


    # TODO: Get odometry data
    def save_odom(self, odom):
        pass

    def create_folders(self):
        if not os.path.exists(self.SAFETY_RGB_PATH):
            print('[!]Creating safety RGB data folder.')
            os.makedirs(self.SAFETY_RGB_PATH)

        if not os.path.exists(self.SAFETY_DEPTH_PATH):
            print('[!]Creating safety DEPTH data folder.')
            os.makedirs(self.SAFETY_DEPTH_PATH)

        if not os.path.exists(self.RGB_PATH):
            print('[!]Creating rgb data folder.')
            os.makedirs(self.RGB_PATH)

        if not os.path.exists(self.DEPTH_PATH):
            print('[!]Creating depth data folder.')
            os.makedirs(self.DEPTH_PATH)

    def record_img(self, data, type, twist):
        try:
            image = self.bridge.imgmsg_to_cv2(data)
            # only record non-zero data
            if self.twist is not None:
                timestamp = data.header.stamp
                v = twist.linear.x
                r = twist.angular.z
                with self.twist_lock:
                    if type == 'depth':
                        filename = self.SAFETY_DEPTH_PATH if self.safety_record else self.DEPTH_PATH
                    else:
                        filename = self.SAFETY_RGB_PATH if self.safety_record else self.RGB_PATH
                    filename = os.path.join(filename, '%s_%s_%s.png' % (timestamp, v, r))
                image = cv2.resize(image, (256, 256))
                # save image
                cv2.imwrite(filename, image)
        except CvBridgeError as error:
            print(error)

    def save_rgb(self, rgb):
        if self.safety_record or self.primary_record or not self.safe:
            self.record_img(rgb, 'rgb', self.twist)

    def save_depth(self, depth):
        if self.primary_record or self.safety_record or not self.safe:
            self.record_img(depth, 'depth', self.twist)

    def get_twist(self, twist):
        with self.twist_lock:
            self.twist = twist

    def get_status(self, joy_cmd):
        self.controller.update(joy_cmd)
        events = self.controller.btn_events
        if 'start_pressed' in events:
            self.primary_record = not self.primary_record
            # disable recording safety
            self.safety_record = False
            if self.primary_record:
                rospy.loginfo('[*]Start recording primary data.')
            else:
                rospy.loginfo('[*]Stop recording primary data.')

        elif 'Triangle_pressed' in events:
            self.safety_record = not self.safety_record
            # disable primary
            self.primary_record = False
            if self.safety_record:
                rospy.loginfo('[*]Start recording safety data.')
            else:
                rospy.loginfo('[*]Stop recording safety data.')

    def reset(self, data):
        """reset the states if crashes"""
        rospy.loginfo(data)

    def update_safety(self, data):
        self.safe = data.data


if __name__ == '__main__':
    try:
        train_iter = sys.argv[2]
        print('[!]Training Iteration %s' % train_iter)
        recorder = Recorder(train_iter)
    except rospy.ROSInterruptException:
        pass
