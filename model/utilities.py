import numpy as np
import cv2
import glob
import pickle
import tensorflow as tf
from model.tf_model import NeuralCommander
import os


def __split_name(labels, name):
    splitted = name.split('_')
    v = float(splitted[-2])
    r = float(splitted[-1].strip('.png'))
    return v, r


def load_data(iteration, val_num=200, threshold=0.99, prototype=2, read_rgb=True, read_depth=False, display=False, safety=False):
    """Read all images in RGB and DEPTH"""
    rgb_imgs = []
    rgb_labels = []
    depth_imgs = []
    depth_labels = []
    filedir = '/media/jxu7/BACK-UP/Data/neural-navigation/prototype-{}/{}'.format(prototype, threshold)
    filedir = os.path.join(filedir, 'safety' if safety else 'primary')
    print(filedir)

    if read_rgb:
        rgb_names = glob.glob(os.path.join(filedir, 'RGB_DATA/%s/*.png' % iteration))
        # remove the unlabeled data
        for n in sorted(rgb_names):
            v, r = __split_name(rgb_labels, n)
            if v == 0.0 and r== 0.0:
                os.remove(n)
                print(n, 'has been removed')
            else:
                rgb_img = cv2.imread(n)
                if display:
                    cv2.imshow('test', rgb_img)
                    cv2.waitKey(3)
                rgb_img = cv2.resize(rgb_img, (128, 128))
                rgb_imgs.append(rgb_img)
                label = np.array([v/0.5, r/4.25])
                rgb_labels.append(label)
        print('[*]Collected %s RGB pictures.' % len(rgb_imgs))

    if read_depth:
        depth_names = glob.glob(os.path.join(filedir, 'DEPTH_DATA/*.png'))
        print('[*]Collected %s DEPTH pictures.' % len(depth_names))
        for n in depth_names:
            __split_name(depth_labels, n)
            depth_img = cv2.imread(n)
            depth_img = cv2.resize(depth_img, (128, 128))
            depth_imgs.append(depth_img)
    return np.array(rgb_imgs), np.array(rgb_labels), np.array(depth_imgs), np.array(depth_labels)


def convert_labels(sess, model, safe_img, reference_label, threshhold, randomize=False, p=None):
    """
        This function labels each image a 0 or a one, where 0 means no danger, 1 means danger.
        Returns the feature extracted from primary policy cnn and the labels.
    """
    safety_features = []
    safety_label = []
    # calculate safety labels and fc1 features
    for i in range(0, reference_label.shape[0]):
        fc1, primary_pi = sess.run([model.layers[-3], model.pi], feed_dict={
            model.x: safe_img[i].reshape(1, 128, 128, 3),
            model.is_training: True
        })
        # if label is 1, it is extremely dangerous, 0 otherwise.
        label = 1 if np.sum(np.square(reference_label[i] - primary_pi[0])) > threshhold else 0

        safety_features.append(fc1)
        safety_label.append(label)
        print('Label %s // Ground Truth %s // Primary Policy %s' % (label, reference_label[i], primary_pi))
        print('[*]Error: %s' % np.sum(np.square(reference_label[i] - primary_pi[0])))
    # labels
    y = np.array(safety_label)
    y = np.expand_dims(y, axis=1)
    # features
    x = np.array(safety_features)
    x = np.squeeze(x, axis=1)
    print('Safe/Unsafe = {}'.format(y[y==0].shape[0]/y[y==1].shape[0]))
    return x, y


def convert_to_pkl(model, sess, train_iter, threshold=0.99):
    """Save tensorflow model to a pickle file"""
    params = {}
    if threshold != 0.99:
        path = '../checkpoint/%s/%s/pkl_model.pkl' % (threshold, train_iter)
    else:
        path = '../checkpoint/%s/pkl_model.pkl' % train_iter
    for v in model.params:
        params[v.name] = sess.run(v)
    with open(path, 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)
    print('[*]Saved to pkl file')
