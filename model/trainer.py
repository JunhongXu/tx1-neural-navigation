import tensorflow as tf
from model.tf_model import NeuralCommander
from model.utilities import *
from tensorflow.contrib.layers import optimize_loss
import numpy as np


RESTORE = True

# check the version of tensorflow
tf_version = tf.__version__
if tf_version.split('.')[0] < 0:
    older = True
else:
    older = False


def train(sess, model, trainer, safety_trainer, num_iter):
    x, y, _, _ = load_data(display=False)
    N, H, W, C = x.shape
    print(x.shape)
    print(y.shape)
    # train the primary policy
    for i in range(num_iter):
        # random index
        index = np.random.randint(0, N, size=128)
        loss, _ = sess.run([model.loss, trainer], feed_dict={model.x: x[index],
                                                             model.y: y[index], model.is_training:True})
        if i % 20 == 0:
            # save
            model.save(sess)
            print(loss)

    safety_x, pi_label, _, _ = load_data(display=False, safety=True)
    safety_label = []
    safety_features = []
    # calculate safety labels and fc1 features
    for i in range(0, pi_label.shape[0]):
        fc1, primary_pi = sess.run([model.layers[-3], model.pi], feed_dict={
            model.x: safety_x[i].reshape(1, 128, 128, 3),
            model.is_training: True
        })
        label = 1 if np.sqrt(np.sum(np.square(pi_label[0] - primary_pi[0]))) > 0.0025 else 0
        safety_features.append(fc1)
        safety_label.append(label)
    # train the safety policy
    for i in range(num_iter):
        pass

if __name__ == '__main__':
    with tf.Session() as sess:
        model = NeuralCommander()
        # trainer for primary policy
        primary_policy_trainer = optimize_loss(model.loss, model.global_pi_setp, learning_rate=0.00001,
                                               optimizer='Adam', variables=[v for v in tf.trainable_variables()
                                                                            if 'cnn' in v.name])

        # trainer for safety policy
        safety_policy_trainer = optimize_loss(model.safety_loss, model.global_safety_step, learning_rate=0.0001,
                                              optimizer='Adam', variables=[v for v in tf.trainable_variables()
                                                                           if 'safety_policy' in v.name])
        if RESTORE:
            model.restore(sess)

        # initialize all variables
        if older:
            sess.run(tf.initialize_all_variables())
        else:
            sess.run(tf.global_variables_initializer())

        train(sess, model, primary_policy_trainer, safety_policy_trainer, 100000)
    convert_to_pkl()


