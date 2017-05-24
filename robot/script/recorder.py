#!/usr/bin/env python
"""
Subscriber:
    1. This node subscribes to the topic /zed/rgb/image_rect_color to save the image using cv2
    2. This node subscribes to the topic /zed/depth/depth_registered to save the depth image using cv2
    3. This node subscribes to the topic /cmd_vel to get filename for each image
    4. This node subscribes to the topic /zed/odom to save the 3D orientation relative to zed_initial_frame


Because safety policy is not able to generalize to unseen environment, meaning when there is a new image that is not
in the training distribution, it will not correctly classify that image to dangerous or safe, which will cause the recorder
not recording dangerous images.

One way to work around of this is to store 5 frames of images and corresponding depth control. If a bumper crash is detected,
store these data into primary/RGB folder.
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
import time
from ca_msgs.msg import Bumper
from collections import deque


class Recorder(object):
    def __init__(self, train_iter, threshold):
        rospy.loginfo('[*]Start recorder.')
        self.record = False
        self.bridge = CvBridge()
        self.train_iter = train_iter
        # create folders
        self.SAFETY_RGB_PATH = '../%s/safety/RGB_DATA/%s' % (threshold, train_iter)
        self.SAFETY_DEPTH_PATH = '../%s/safety/DEPTH_DATA/%s' % (threshold, train_iter)
        self.DEPTH_PATH = '../%s/primary/DEPTH_DATA/%s' % (threshold, train_iter)
        self.RGB_PATH = '../%s/primary/RGB_DATA/%s' % (threshold, train_iter)
        self.create_folders()
        # self.twist_lock = Lock()
        self.controller = PS3()
        self.primary_record = False
        self.safety_record = False
        self.safe = True
        self.twist = Twist()
        self.depth_twist = Twist()
        self.avoided = True
        self.start_time = 0
        self.end_time = 0
        self.data_name = 'Iter_%s' % train_iter
        self.distance_travelled = 0
        self.curr_x = self.previous_x = 0.0
        self.total_speed = 0
        self.neural_net_on = False
        self.human_on = False
        self.depth_on = False
        self.num_crashes = 0
        # pre-stored frames and controls
        self.stored_data = deque(maxlen=8)
        self.bumper_lock = Lock()

        rospy.Subscriber('/zed/rgb/image_rect_color', Image, self.save_rgb)

        # cmd_vel
        rospy.Subscriber('/cmd_vel', Twist, self.get_twist)
        # odom
        rospy.Subscriber('/odom', Odometry, self.save_odom)
        # joy
        rospy.Subscriber('/joy', Joy, self.get_status)
        # safety
        rospy.Subscriber('/safety', Bool, self.update_safety)

        rospy.Subscriber('/reset', Bool, self.reset)

        rospy.Subscriber('/depth_control', Twist, self.update_depth_control)

        rospy.Subscriber('/bumper', Bumper, self.bumper)

        rospy.Subscriber('/toggle_nn', Bool, self.is_nn_on)

        rospy.Subscriber('/toggle_human', Bool, self.is_human_on)

        rospy.Subscriber('/toggle_depth', Bool, self.is_depth_on)

        rospy.init_node('recorder')

        rospy.on_shutdown(self.shutdown)
        # keeps the node alive
        rospy.spin()

    def is_human_on(self, data):
        if data.data is True:
            rospy.loginfo('[*]Human Testing...')
            self.data_name = 'Human'
            self.start_time = time.time()

    def is_depth_on(self, data):
        if data.data is True:
            rospy.loginfo('[*]Depth Testing...')
            self.data_name = 'Depth'
            self.start_time = time.time()

    def is_nn_on(self, data):
        if data.data is True:
            rospy.loginfo('[*]Recorder: Neural Network ON')
            self.neural_net_on = True
            self.start_time = time.time()
        else:
            rospy.loginfo('[*]Recorder: Neural Network ON')
            self.neural_net_on = False

    def bumper(self, data):
        if self.neural_net_on:
            if data.is_left_pressed or data.is_right_pressed:
                self.end_time = time.time()
                # store images
                self.num_crashes += 1
                rospy.loginfo('[*]Saving bumper images')
                with self.bumper_lock:
                    for idx, (timestamp, control, img) in enumerate(self.stored_data):

                        v = (-1/8)*(idx+1) + 1
                        print(v, idx)
                        r = control.angular.z
                        print(v, r)
                        filename = self.RGB_PATH
                        filename = os.path.join(filename, '%s_%s_%s.png' % (timestamp, v, r))
                        cv2.imwrite(filename, img)

    def shutdown(self):
        crashes = '{}, {}'.format(self.train_iter, self.num_crashes)

        rospy.loginfo('[*]Saving data...')
        if not os.path.exists('../crashes.csv'):
            with open('../crashes.csv', 'w') as f:
                f.write(crashes)
        else:
            with open('../crashes.csv', 'a') as f:
                f.write('\n{}'.format(crashes))

        data = '{}, {}, {}'.format(self.data_name, self.distance_travelled, self.end_time - self.start_time)
        if not os.path.exists('../data.csv'):
            with open('../data.csv', 'w') as f:
                f.write('name,dist,time')
                f.write('\n{}'.format(data))
        else:
            with open('../data.csv', 'a') as f:
                f.write('\n{}'.format(crashes))

    def update_depth_control(self, data):
        self.depth_twist = data

    # TODO: Get odometry data
    def save_odom(self, odom):
        pose = odom.pose.pose
        x = pose.position.x
        distance = abs(x - self.previous_x)
        self.previous_x = x
        self.distance_travelled += distance

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
            image = cv2.resize(image, (256, 256))
            # only record non-zero data
            if self.twist is not None:
                timestamp = data.header.stamp
                v = twist.linear.x
                r = twist.angular.z
                with self.bumper_lock:
                    self.stored_data.append((timestamp, twist, image))

                # with self.twist_lock:
                if type == 'depth':
                    filename = self.SAFETY_DEPTH_PATH if self.safety_record else self.DEPTH_PATH
                else:
                    filename = self.SAFETY_RGB_PATH if self.safety_record else self.RGB_PATH
                filename = os.path.join(filename, '%s_%s_%s.png' % (timestamp, v, r))
                # save image
                if type == 'rgb':
                    cv2.imwrite(filename, image)
        except CvBridgeError as error:
            print(error)

    def save_rgb(self, rgb):
        # for bumper
        self.record_img(rgb, 'bumper', self.depth_twist)
        if self.safety_record or self.primary_record:
            self.record_img(rgb, 'rgb', self.twist)
        elif self.neural_net_on:
            if not self.safe and self.avoided:
                self.record_img(rgb, 'rgb', self.depth_twist)

    def get_twist(self, twist):
        # with self.twist_lock:
        self.twist = twist
        self.total_speed += twist.linear.x

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
        self.avoided = data.data

    def update_safety(self, data):
        self.safe = data.data

if __name__ == '__main__':
    try:
        train_iter = sys.argv[2]
        threshold = sys.argv[4]
        print('[!]Training Iteration %s' % train_iter)
        print('[!]Threshold %s' % threshold)
        recorder = Recorder(train_iter, threshold)
    except rospy.ROSInterruptException:
        pass
