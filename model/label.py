"""
This module is for labelling the images by human
"""
import cv2
import numpy as np
import glob
import os


class Label(object):
    def __init__(self, iteration):
        self.names = self.read_names(iteration)
        self.total_pic = len(self.names)
        self.curr_idx = 0
        self.curr_name = self.names[self.curr_idx]
        self.curr_img = cv2.imread(self.curr_name)
        self.curr_v = 0
        self.curr_r = 0
        self.new_name = None
        self.quit = False
        self.img_dim = 256
        self.speed_mode = False
        self.firs_loaded = True

    def on_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.speed_mode:
                cv2.line(self.curr_img, (0, y), (self.img_dim, y), (255, 10, 150), thickness=2)
                self.curr_v = -0.5/(self.img_dim/2) * y + 0.5
            else:
                cv2.line(self.curr_img, (x, 0), (x, self.img_dim), (150, 10, 255), thickness=2)
                self.curr_r = -4.5/(self.img_dim/2) * x + 4.5
            print('Current speed %s_%s' %(self.curr_v, self.curr_r))
        cv2.imshow('image', self.curr_img)

    def read_names (self, iteration):
        # read all image names
        filedir = '/media/jxu7/BACK-UP/Data/neural-navigation/iteration_%s/primary' % iteration
        names = glob.glob(os.path.join(filedir, '*.png'))
        print('Found %s images' % len(names))
        return names

    def parse_name(self):
        splitted = self.curr_name.split('_')
        v = float(splitted[-2])
        r = float(splitted[-1].strip('.png'))
        return v, r

    def x_tf(self, coord):
        """opencv coordinate system has a zero point at upper left, this transforms it to lower left"""
        x = coord[0]
        y = self.img_dim - coord[1]
        return x, y

    def keyboard_event(self, event):
        if event == ord('a'):
            self.curr_idx -= 1
            if abs(self.curr_idx) == self.total_pic:
                self.curr_idx=0
            # plot the previous image
            self.show()
        elif event == ord('d'):
            self.curr_idx += 1
            if self.curr_idx == self.total_pic:
                self.curr_idx = 0
            self.show()
        elif event == ord('v'):
            self.speed_mode = not self.speed_mode
        elif event == ord('r'):
            self.show()
        elif event == ord('s'):
            # rename the current image file
            name = self.curr_name.split('/')
            base_path = name[1:-1]
            img_name = name[-1].split('_')
            base_path.append(str('%s_%s_%s.png' % (img_name[0], self.curr_v, self.curr_r)))
            new_name = os.path.join(*base_path)
            new_name = '/'+new_name
            os.rename(self.curr_name, new_name)
            self.names.pop(self.curr_idx)
            self.curr_idx += 1
            print('[*]File saved as %s' % new_name)
            self.show()
        elif event == 27:
            cv2.destroyAllWindows()
            self.quit = True

    def draw_line(self, orientation, color, coord=None, needs_transform=False):
        if needs_transform:
            if orientation == 'H':
                line = ((0, coord), (self.img_dim, coord))
                cv2.line(self.curr_img, self.x_tf(line[0]), self.x_tf(line[1]), color, thickness=2)
            elif orientation == 'V':
                line = ((coord, 0), (coord, self.img_dim))
                cv2.line(self.curr_img, self.x_tf(line[0]), self.x_tf(line[1]), color, thickness=2)

        else:
            if orientation == 'H':
                y = self.img_dim//2
                line = ((0, y), (self.img_dim, y))
                cv2.line(self.curr_img, self.x_tf(line[0]), self.x_tf(line[1]), color, thickness=1)
            elif orientation == 'V':
                x = self.img_dim//2
                line = ((x, 0), (x, self.img_dim))
                cv2.line(self.curr_img, self.x_tf(line[0]), self.x_tf(line[1]), color, thickness=1)

    def plot(self):
        # parse the name
        self.speed_mode = False
        v, r = self.parse_name()
        self.curr_v = v
        self.curr_r = r
        # horizontal line for velocity
        y = self.img_dim//2 + int(self.img_dim/2/0.5 * v)
        self.draw_line('H', (255, 0, 0), coord=y, needs_transform=True)

        # vertical line for rotation
        x = self.img_dim//2 + int(self.img_dim/2/4.5 * -r)
        self.draw_line( 'V', (0, 0, 255), coord=x, needs_transform=True)

        # draw the center line
        self.draw_line(color=(0, 125, 0), orientation='H')
        self.draw_line(color=(0, 125, 0), orientation='V')

    def show(self):
        print('[!]There are %s left to be labelled' % len(self.names))
        if len(self.names) == 0:
            self.quit = True
        self.curr_name = self.names[self.curr_idx]
        self.curr_img = cv2.imread(self.curr_name)
        print('[!]Current name: %s' % self.curr_name)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.on_mouse)
        self.plot()
        cv2.imshow('image', self.curr_img)

    def change_labels(self):
        """Label all the images"""
        while 1:
            if self.firs_loaded:
                self.show()
                self.firs_loaded = False
            self.keyboard_event(cv2.waitKey(0))
            if self.quit:
                break



if __name__ == '__main__':
    label = Label(1)
    label.change_labels()
