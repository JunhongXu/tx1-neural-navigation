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
            3: 'start',
            4: 'ArrowUp', 5: 'ArrowRight', 6: 'ArrowDown', 7: 'ArrowLeft',
            8: 'L2', 9: 'R2', 10: 'L1', 11: 'R1',
            12: 'Triangle', 13: 'Circle', 14: 'X', 15: 'Square'
        }

        self.btn_idx = {v: k for k, v in self.idx_btn.iteritems()}
        self.btn_state_prev = {k: False for k in self.btn_idx.keys()}
        self.btn_state_curr = copy.deepcopy(self.btn_state_prev)
        self.left_stick = 0.0
        self.right_stick = 0.0

    def update(self, joy_data):
        left_stick_y = joy_data.axes[1]
        right_stick_x = joy_data.axes[2]
        self.left_stick = left_stick_y
        self.right_stick = right_stick_x
        self.btn_events = self._get_btn_events(joy_data)

    def _get_btn_events(self, data):
        btn_evts = []
        for i in range(3, 16):
            self.btn_state_prev[self.idx_btn[i]] = self.btn_state_curr[self.idx_btn[i]]
            self.btn_state_curr[self.idx_btn[i]] = data.buttons[i] == 1
        for i in range(3, 16):
            if self.btn_state_curr[self.idx_btn[i]] != self.btn_state_prev[self.idx_btn[i]]:
                if self.btn_state_curr[self.idx_btn[i]]:
                    btn_evts.append(self.idx_btn[i] + "_pressed")
                else:
                    btn_evts.append(self.idx_btn[i] + "_released")
        return btn_evts
