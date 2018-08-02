import tensorflow as tf
from dataset import DataSet
import os


class LstmWithSvd:

    def __init__(self, input_size: int, map_size: int, hidden_unit: int, output_size: int, time_step: int,
                 batch_size: int = 1):
        self.map_size = map_size
        self.time_step = time_step
        self.hidden_unit = hidden_unit
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, shape=[batch_size, time_step, input_size], name='input_data')
        self.y = tf.placeholder(tf.float32, shape=[batch_size, time_step, output_size], name='train_data')

        with tf.name_scope('input_mapping'):
            map_mtx = tf.Variable(tf.constant(0.0, shape=[map_size, input_size]), trainable=False)
            self.map_saver = tf.train.Saver({'map_mtx': map_mtx})
            # Reshape tensor to 2-dimension for matrix multiplication
            x = tf.reshape(self.x, [-1, input_size])
            input_map = x @ tf.transpose(map_mtx)
        with tf.name_scope('input_layer'):
            with tf.name_scope('weights'):
                w_in = tf.Variable(tf.random_normal([map_size, hidden_unit]))
                tf.summary.histogram('weights', w_in)
            with tf.name_scope('biases'):
                b_in = tf.Variable(tf.constant(0.1, shape=[hidden_unit, ]))
                tf.summary.histogram('biases', b_in)
            # linear map
            with tf.name_scope('full_connection'):
                input_rnn = input_map @ w_in + b_in
                # Reshape tensor back to 3-dimension
                input_rnn = tf.reshape(input_rnn, [-1, time_step, hidden_unit])
        with tf.name_scope('lstm_units'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_unit)
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            # output_rnn: results in each step
            # last_states: final states
            output_rnn, last_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
            # summary
            kernel_vals, bias_vals = cell.variables
            tf.summary.histogram('kernel', kernel_vals)
            tf.summary.histogram('bias', bias_vals)
        with tf.name_scope('output_layer'):
            with tf.name_scope('weights'):
                w_out = tf.Variable(tf.random_normal([hidden_unit, map_size]))
                tf.summary.histogram('weights', w_out)
            with tf.name_scope('biases'):
                b_out = tf.Variable(tf.constant(0.1, shape=[map_size, ]))
                tf.summary.histogram('biases', b_out)
            with tf.name_scope('full_connection'):
                output_rnn = tf.reshape(output_rnn, [-1, hidden_unit])
                output_map = output_rnn @ w_out + b_out
        with tf.name_scope('output_mapping'):
            output_ = output_map @ map_mtx
            output_ = tf.reshape(output_, [-1, time_step, output_size])
        with tf.name_scope('softmax'):
            self.prd = tf.nn.softmax(output_)

    def train(self, user_limit, start_learning_rate, training_steps, decay_rate):
        # data set
        train_x, train_y = DataSet(user_limit, self.time_step).lstm_train()

        # error and optimize function
        with tf.name_scope('train'):
            error = tf.reduce_mean(tf.abs(self.prd - self.y))
            tf.summary.scalar('error', error)
            # Dynamic learning rate
            global_step = tf.placeholder(tf.int16, name='global_step')
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, training_steps, decay_rate)
            tf.summary.scalar('learning_rate', learning_rate)
            update_op = tf.train.AdamOptimizer(learning_rate).minimize(error)

        # Run session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Merge summaries
            merged = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('saved_models/LSTM_with_SVD_%d/log' % self.hidden_unit, sess.graph)

            # Initialize global variables
            sess.run(tf.global_variables_initializer())

            # Initialize mapping matrix
            map_file = tf.train.latest_checkpoint('saved_models/Prl_SVD_%d/' % self.map_size)
            self.map_saver.restore(sess, map_file)

            # Start learning
            err_sum = 0
            turns = 0
            data_len = len(train_x)
            for start in range(0, data_len * training_steps, self.batch_size):
                end = start + self.batch_size
                curr_step = start // data_len

                if curr_step == end // data_len:
                    feed_dict = {
                        self.x: train_x[start % data_len:end % data_len],
                        self.y: train_y[start % data_len:end % data_len],
                        global_step: curr_step
                    }
                    _, curr_err, _ = sess.run([update_op, error, learning_rate], feed_dict=feed_dict)
                    err_sum += curr_err
                    turns += 1
                else:
                    feed_dict = {
                        self.x: train_x[start % data_len:] + train_x[:end % data_len],
                        self.y: train_y[start % data_len:] + train_y[:end % data_len],
                        global_step: curr_step
                    }
                    _, curr_err, curr_lr, summary = sess.run([update_op, error, learning_rate, merged],
                                                             feed_dict=feed_dict)
                    err_sum += curr_err
                    turns += 1

                    # Write summaries
                    summary_writer.add_summary(summary, global_step=curr_step)

                    print('Step %d: error = %g, learning rate = %g' % (curr_step, err_sum / turns, curr_lr))
                    err_sum = 0
                    turns = 0

            # Save model
            saver = tf.train.Saver()
            saver.save(sess, 'saved_models/LSTM_with_SVD_%d/model' % self.hidden_unit, global_step=training_steps)

    def evaluate(self, user_limit):
        dataset = DataSet(user_limit, self.time_step)
        test_x = dataset.lstm_test()
        test_y, known = dataset.correct_data()

        hits = 0
        crrs = 0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            module_file = tf.train.latest_checkpoint('saved_models/LSTM_with_SVD_%d/' % self.hidden_unit)
            saver = tf.train.Saver()
            saver.restore(sess, module_file)

            for i in range(len(test_x)):
                next_ = sess.run(self.prd, feed_dict={self.x: [test_x[i]]})

                priority = list(zip(next_[0][-1], dataset.game_tpl))
                priority.sort(reverse=True)
                priority = list(zip(*priority))[1]

                rec = [g for g in priority if g not in known[i]][:50]
                crr = [h for h in test_y[i] if h not in known[i]]

                hits += len(set(rec).intersection(set(crr)))
                crrs += len(crr)

                print('[hits, corrs] = [%d, %d]' % (hits, crrs))

        print('recall = %g' % (hits / crrs))


if __name__ == '__main__':
    try:
        model = LstmWithSvd(input_size=7649, map_size=256, hidden_unit=256, output_size=7649, time_step=8,
                            batch_size=128)
        model.train(user_limit=2000, start_learning_rate=0.001, training_steps=1000, decay_rate=0.05)

        # model = LstmWithSvd(input_size=7649, hidden_unit=256, output_size=7649, time_step=8)
        # model.evaluate(user_limit=2000)
    finally:
        os.system('shutdown /s /t 60')
