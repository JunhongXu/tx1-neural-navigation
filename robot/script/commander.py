#!/usr/bin/env python
"""
Subscriber:
    1. Commander node subscribes to /joy topic published by the host pc and listens to the command of joystick and
       send commands to /cmd_vel.
    2. Commander node subscribes to /neural_command topic published by neural_commander node to send commands to
       /cmd_vel
Publisher:
    1. Commander node publishes Twist to /cmd_vel topic to control the icreate robot
"""


import rospy
from sensor_msgs.msg import Joy
from ca_msgs.msg import Bumper
from geometry_msgs.msg import TwistStamped, Twist
from controller import *
from nav_msgs.msg import Odometry
import tf.transformations as transformations
import numpy as np


class Commander(object):
    def __init__(self):
        rospy.loginfo('[*]Start commander.')
        # joystick
        self.joycmd = Twist()
        self.nn_cmd = Twist()
        self.bumper_cmd = Twist()
        # mode
        self.neuralnet_mode = False
        self.ps3 = PS3()

        self.euler = 0
        self.desired_euler = 0
        self.bumper = False
        self.curr_x = 0
        self.prev_x = 0
        self.x = 0
        self.is_avoid = True

        # subscriber
        rospy.Subscriber('/joy', Joy, self.joystick_cmd, queue_size=5)
        rospy.Subscriber('/neural_cmd', Twist, self.neural_cmd, queue_size=5)
        rospy.Subscriber('/bumper', Bumper, self.reset)
        rospy.Subscriber('/odom', Odometry, self.update_odom)
        # publisher
        self.move_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        # initialize rosnode
        rospy.init_node('commander', anonymous=True)
        self.rate = rospy.Rate(60)

    def run(self):
        while not rospy.is_shutdown():
            self.send_cmd()
            self.rate.sleep()

    def reset(self, data):
        """reset initial state"""
        if data.is_left_pressed and data.is_right_pressed:
            # self.curr_x = self.prev_x = self.x
            self.bumper = True
            # reverse if encounter obstacles on both sides
            self.desired_euler  = -np.pi
        elif data.is_left_pressed:
            self.bumper = True
            self.desired_euler = 0
        elif data.is_right_pressed:
            self.bumper = True
            self.desired_euler = 0
        else:
            self.bumper = False

    def update_odom(self, data):
        pose = data.pose.pose
        quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        position = pose.position.x
        euler = transformations.euler_from_quaternion(quaternion)
        # self.euler = euler[-1]
        if self.bumper:
            self.is_avoid = False
            linear_avoid = False
            angular_avoid = False
            if np.abs(self.curr_x - self.prev_x) < 0.5:
                self.bumper_cmd.linear.x = -0.5
                self.curr_x = position
            else:
                linear_avoid = True

            if self.desired_euler < euler[-1]:
                # self.bumper_cmd.linear.x =
                pass
            else:
                angular_avoid = True

            if linear_avoid:
                self.is_avoid = True

            rospy.loginfo('[!]Bumper')
        else:
            self.curr_x = self.prev_x = position

    def joystick_cmd(self, cmd):
        """joystick command, -0.5<=linear.x<=0.5; -4.25<=auglar.z<=4.25"""
        self.ps3.update(cmd)
        vel = self.ps3.left_stick*0.5
        angular = self.ps3.right_stick * 4.25
        self.joycmd.angular.z = angular
        self.joycmd.linear.x = vel

        btn_events = self.ps3.btn_events
        if 'R2_pressed' in btn_events:
            self.neuralnet_mode = not self.neuralnet_mode
            if self.neuralnet_mode:
                rospy.loginfo('[*]Neural network controlling...')
            else:
                rospy.loginfo('[*]Human controlling...')

    def neural_cmd(self, cmd):
        self.nn_cmd = cmd

    def send_cmd(self):
        if self.neuralnet_mode:
            self.move_pub.publish(self.nn_cmd)
        elif self.is_avoid:
            self.move_pub.publish(self.bumper_cmd)
        else:
            self.move_pub.publish(self.joycmd)


if __name__ == '__main__':
    try:
        commander = Commander()
        commander.run()
    except rospy.ROSInterruptException:
        pass
