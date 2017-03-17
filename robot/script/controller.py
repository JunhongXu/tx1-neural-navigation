#!/usr/bin/env python
"""
A ps3 specific controller module to detect keypress, keyrelease, joystick movements

1. joystick left: joy.axes[0] horizontal, joy.axes[1] vertical
2. joystick right: joy.axes[2] horizontal joy.axes[3] vertical
3. [start_btn]: joy.button[3]
4. [up]: joy.button[4]/[right]: joy.button[5]/[down]: joy.button[6]/left: joy.button[7]
5. [L2]: joy.button[8]/[R2]: joy.button[9]/[L1]: joy.button[10]/[R1]: joy.button[11]
6. [triangle]: joy.button[12]/[circle]: joy.button[13]/[x]: joy.button[14]/[square]: joy.button[15]
"""
import numpy as np
import copy


class PS3(object):
    def __init__(self):
        self.idx_btn = {

        }
        self.btn_idx = {

        }
        # a dictionary of buttons. {btn_name:True/False}
        self.buttons = {
            'left': False,
        }
