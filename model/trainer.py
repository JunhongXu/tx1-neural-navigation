import tensorflow as tf
from model.tf_model import NeuralCommander
from model.utilities import *
from tensorflow.contrib.layers import optimize_loss
import numpy as np


TRAIN_ITER = 0
BATCH_SIZE = 128
SAFETY_THRESHOLD = 0.0025
DISPLAY = False
NUM_ITERS = 20000


def train_primary_policy(x, y, writer, model, num_iter, trainer):
    N, H, W, C = x.shape
    summary = tf.summary.scalar('primary_loss', model.loss)
    # train the primary policy
    for i in range(num_iter):
        # random index
        index = np.random.randint(0, N, size=BATCH_SIZE)
        loss, _, loss_summary = sess.run([model.loss, trainer, summary], feed_dict={model.x: x[index],
                                                             model.y: y[index], model.is_training:True})
        writer.add_summary(loss_summary, global_step=i)
        if i % 99 == 0:
            # save
            model.save(sess, TRAIN_ITER)
            print('[*]At iteration %s/%s, loss is %s' % (i, num_iter, loss))


def train_safety_policy(safety_x, pi_label, writer, model, num_iter, trainer):

    N, H, W, C = safety_x.shape
    summary = tf.summary.scalar('safety_loss', model.safety_loss)
    safety_label = []
    safety_features = []
    # calculate safety labels and fc1 features
    for i in range(0, pi_label.shape[0]):
        fc1, primary_pi = sess.run([model.layers[-3], model.pi], feed_dict={
            model.x: safety_x[i].reshape(1, 128, 128, 3),
            model.is_training: True
        })
        # if label is 1, it is extremely dangerous, 0 otherwise.
        label = 1 if np.sum(np.square(pi_label[i] - primary_pi[0])) > SAFETY_THRESHOLD else 0

        safety_features.append(fc1)
        safety_label.append(label)
        print('Label %s // Ground Truth %s // Primary Policy %s' % (label, pi_label[i], primary_pi))
        print('[*]Error: %s' % np.sum(np.square(pi_label[i] - primary_pi[0])))

    y = np.array(safety_label)
    y = np.expand_dims(y, axis=1)
    x = np.array(safety_features)
    x = np.squeeze(x, axis=1)
    # train the safety policy
    for i in range(num_iter):
        # random index
        index = np.random.randint(0, N, size=BATCH_SIZE)
        loss, _, loss_summary = sess.run([model.safety_loss, trainer, summary], feed_dict={model.safety_inpt: x[index],
                                                             model.safety_y: y[index], model.is_training:True})
        writer.add_summary(loss_summary, global_step=i)
        if i % 99 == 0:
            # save
            model.save(sess, TRAIN_ITER)
            print('[*]At iteration %s/%s, loss is %s' % (i, num_iter, loss))


def train(sess, model, trainer, safety_trainer, num_iter):
    model_save_path = os.path.join('../checkpoint', str(TRAIN_ITER))
    summary_save_path = os.path.join('../summary', str(TRAIN_ITER))
    # create folders
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if not os.path.exists(summary_save_path):
        os.makedirs(summary_save_path)

    # load data for primary policy
    x, y, _, _ = load_data(TRAIN_ITER, display=DISPLAY)

    # load data for training safety policy
    safety_x, pi_label, _, _ = load_data(TRAIN_ITER, display=DISPLAY, safety=True)

    # concate dataset
    safety_x = np.concatenate((safety_x, x), axis=0)
    pi_label = np.concatenate((pi_label, y), axis=0)

    # summary writer
    writer = tf.summary.FileWriter(summary_save_path, graph=sess.graph)

    # primary policy
    train_primary_policy(x, y, writer, num_iter=num_iter, trainer=trainer, model=model)

    # safety policy
    train_safety_policy(safety_x, pi_label, writer, model, num_iter, safety_trainer)


if __name__ == '__main__':
    with tf.Session() as sess:
        model = NeuralCommander()
        # trainer for primary policy
        primary_policy_trainer = optimize_loss(
            model.loss, model.global_pi_setp, learning_rate=0.00001, name='primary_optimizer',
            optimizer='Adam', variables=[v for v in tf.trainable_variables() if 'cnn' in v.name]
        )

        # trainer for safety policy
        safety_policy_trainer = optimize_loss(
            model.safety_loss, model.global_safety_step, learning_rate=0.0003, name='safety_optimizer',
            optimizer='Adam', variables=[v for v in tf.trainable_variables() if 'safety_policy' in v.name]
        )

        # initialize all variables
        sess.run(tf.global_variables_initializer())

        if TRAIN_ITER > -1:
            model.restore(sess, TRAIN_ITER)

        train(sess, model, primary_policy_trainer, safety_policy_trainer, NUM_ITERS)
    convert_to_pkl(TRAIN_ITER)


