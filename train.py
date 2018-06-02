import tensorflow as tf
import os
from dataset import DataSet
from models import Models

# global parameters
user_limit = 2000


def lstm_train(hidden_unit: int):
    # LSTM parameters
    input_size = output_size = 7649
    time_step = 8
    batch_size = 128

    # training parameters
    start_learning_rate = 0.001
    decay_rate = 0.05
    training_steps = 500
    global_step = tf.Variable(0, trainable=False)

    # placeholder and data set
    x = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    train_x, train_y = DataSet(user_limit, time_step).lstm_train()

    # Load model
    prd = Models.lstm(x, input_size, hidden_unit, output_size, time_step, batch_size)

    # error and optimize function
    with tf.name_scope('train'):
        error = tf.reduce_mean(tf.abs(tf.subtract(prd, y)))
        tf.summary.scalar('error', error)
        # Dynamic learning rate
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, training_steps, decay_rate)
        tf.summary.scalar('learning_rate', learning_rate)
        update_op = tf.train.AdamOptimizer(learning_rate).minimize(error)

    # initialize or reload model
    saver = tf.train.Saver(tf.global_variables())
    # module_file = tf.train.latest_checkpoint()

    # Run session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Merge summaries
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('models/NoCodec_%d/log' % hidden_unit, sess.graph)

        # initialize or reload model
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)

        # Start learning
        err_sum = 0
        turns = 0
        data_len = len(train_x)
        for start in range(0, data_len * training_steps, batch_size):
            end = start + batch_size
            curr_step = start // data_len

            if curr_step == end // data_len:
                feed_dict = {
                    x: train_x[start % data_len:end % data_len],
                    y: train_y[start % data_len:end % data_len],
                    global_step: curr_step
                }
                _, curr_err, _ = sess.run([update_op, error, learning_rate], feed_dict=feed_dict)
                err_sum += curr_err
                turns += 1
            else:
                feed_dict = {
                    x: train_x[start % data_len:] + train_x[:end % data_len],
                    y: train_y[start % data_len:] + train_y[:end % data_len],
                    global_step: curr_step
                }
                _, curr_err, curr_lr = sess.run([update_op, error, learning_rate], feed_dict=feed_dict)
                err_sum += curr_err
                turns += 1

                # Write summaries
                summary = sess.run(merged, feed_dict=feed_dict)
                summary_writer.add_summary(summary, global_step=curr_step)

                print('Step %d: current error = %g, current learning rate = %g' % (
                    curr_step, err_sum / turns, curr_lr))
                err_sum = 0
                turns = 0

        # Save model
        saver.save(sess, 'models/NoSVD_%d/NoCodec.model' % hidden_unit, global_step=training_steps)


def codec_train(hidden_unit: int):
    # codec parameters
    input_size = 7649

    # training parameters
    start_learning_rate = 0.001
    decay_rate = 0.3
    training_steps = 100
    global_step = tf.Variable(0, trainable=False)

    # placeholder and data set
    x = tf.placeholder(tf.float32, shape=[None, input_size])
    y = tf.placeholder(tf.float32, shape=[None, input_size])
    train = DataSet(user_limit).codec_train()

    # Load model
    _, prd = Models.codec(x, input_size, hidden_unit)

    # error and optimize function
    with tf.name_scope('train'):
        error = tf.reduce_mean(tf.abs(tf.subtract(prd, y)))
        tf.summary.scalar('error', error)
        # Dynamic learning rate
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, training_steps, decay_rate)
        tf.summary.scalar('learning_rate', learning_rate)
        update_op = tf.train.AdamOptimizer(learning_rate).minimize(error)

    # initialize or reload model
    saver = tf.train.Saver(tf.global_variables())
    # module_file = tf.train.latest_checkpoint()

    # Run session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Merge summaries
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('models/Codec_%d/log' % hidden_unit, sess.graph)

        # initialize or reload model
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)

        # Start learning
        for turn in range(training_steps):
            feed_dict = {}
            err_sum = 0
            curr_lr = 0

            for input_ in train:
                feed_dict = {x: input_, y: input_, global_step: turn}
                _, curr_err, curr_lr = sess.run([update_op, error, learning_rate], feed_dict=feed_dict)
                err_sum += curr_err

            # Write summaries
            summary = sess.run(merged, feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step=turn)

            print('Step %d: current error = %g, current learning rate = %g' % (turn, err_sum / len(train), curr_lr))

        # Save model
        saver.save(sess, 'models/Codec_%d/Codec.model' % hidden_unit, global_step=training_steps)


if __name__ == '__main__':
    try:
        codec_train(256)
    finally:
        # os.system('shutdown /s /t 60')
        pass
