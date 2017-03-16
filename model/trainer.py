import tensorflow as tf
from model.tf_model import NeuralCommander
from model.utilities import *
from tensorflow.contrib.layers import optimize_loss
import numpy as np


# check the version of tensorflow
tf_version = tf.__version__
if tf_version.split('.')[0] < 0:
    older = True
else:
    older = False


def train(sess, model, trainer, num_iter):
    x, y, _, _ = load_data()
    N, H, W, C = x.shape
    print(x.shape)
    print(y.shape)
    for i in range(num_iter):
        # random index
        index = np.random.randint(0, N, size=128)
        loss, _ = sess.run([model.loss, trainer], feed_dict={model.x: x[index],
                                                             model.y: y[index], model.is_training:True})
        if i % 20 == 0:
            # save
            model.save(sess)
            print(loss)

if __name__ == '__main__':
    with tf.Session() as sess:
        model = NeuralCommander()
        trainer = optimize_loss(model.loss, model.global_setp, learning_rate=0.00001, optimizer='Adam')
        # initialize all variables
        if older:
            sess.run(tf.initialize_all_variables())
        else:
            sess.run(tf.global_variables_initializer())

        train(sess, model, trainer, 100000)
    convert_to_pkl()


