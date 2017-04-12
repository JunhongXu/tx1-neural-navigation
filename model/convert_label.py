import tensorflow as tf
import cv2
from utilities import *
from tf_model import NeuralCommander


if __name__ == '__main__':
    # load safety data
    x, y, _, _ = load_data(0, display=False)

    # load data for training safety policy
    safety_x, pi_label, _, _ = load_data(0, display=False, safety=True)

    # concate dataset
    safety_x = np.concatenate((safety_x, x), axis=0)
    pi_label = np.concatenate((pi_label, y), axis=0)

    with tf.Session() as sess:
        model = NeuralCommander()
        model.restore(sess, 0)
        x, y = convert_labels(sess=sess, model=model, reference_label=pi_label, safe_img=safety_x, threshhold=0.0025)

    # save
    safe_idx = 0
    unsafe_idx = 0
    for image, label in zip(safety_x, y):
        # safe
        if label[0] == 0:
            cv2.imwrite('/home/jxu7/Research/WGAN-TF/data/indoor-0/%s.png' % safe_idx, image)
            safe_idx += 1

        # unsafe
        else:
            cv2.imwrite('/home/jxu7/Research/WGAN-TF/data/indoor-1/%s.png' % unsafe_idx, image)
            unsafe_idx += 1
