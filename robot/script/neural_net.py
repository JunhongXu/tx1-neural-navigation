#!/usr/bin/env python

"""
NeuralNet subscribes to /zed/rgb/image_rect_color and take the images to make predictions.
NeuralNet publishes to 'neural_cmd' topic which is being subscribed by commander to change the twist.
"""

import rospy
from sensor_msgs.msg import Image
from model.tf_model import NeuralCommander
import tensorflow as tf


class NeuralNet(object):
    def __init__(self):
        pass
