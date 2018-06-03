import tensorflow as tf
from dataset import DataSet


class LSTM:

    def __init__(self, input_size: int, hidden_unit: int, output_size: int, time_step: int, batch_size: int = 1):
        self.time_step = time_step
        self.hidden_unit = hidden_unit
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, shape=[batch_size, time_step, input_size])
        self.y = tf.placeholder(tf.float32, shape=[batch_size, time_step, output_size])

        with tf.name_scope('input_layer'):
            with tf.name_scope('weights'):
                w_in = tf.Variable(tf.random_normal([input_size, hidden_unit]))
                tf.summary.histogram('weights', w_in)
            with tf.name_scope('biases'):
                b_in = tf.Variable(tf.constant(0.1, shape=[hidden_unit, ]))
                tf.summary.histogram('biases', b_in)
            with tf.name_scope('full_connection'):
                # Reshape tensor to 2-dimension for matrix multiplication
                # (b, s, i) => (b* s, i)
                input_ = tf.reshape(self.x, [-1, input_size])
                # linear map
                # (b* s, i) => (b* s, h)
                input_rnn = tf.matmul(input_, w_in) + b_in
                # Reshape tensor back to 3-dimension
                # (b* s, h) => (b, s, h)
                input_rnn = tf.reshape(input_rnn, [-1, time_step, hidden_unit])
        with tf.name_scope('lstm_units'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_unit)
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            # output_rnn: results in each step
            # last_states: final states
            output_rnn, last_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
        with tf.name_scope('output_layer'):
            with tf.name_scope('weights'):
                w_out = tf.Variable(tf.random_normal([hidden_unit, output_size]))
                tf.summary.histogram('weights', w_out)
            with tf.name_scope('biases'):
                b_out = tf.Variable(tf.constant(0.1, shape=[output_size, ]))
                tf.summary.histogram('biases', b_out)
            with tf.name_scope('full_connection'):
                # (b, s, h) => (b* s, h)
                output_rnn = tf.reshape(output_rnn, [-1, hidden_unit])
                # (b* s, h) => (b* s, o)
                output_ = tf.matmul(output_rnn, w_out) + b_out
        with tf.name_scope('softmax'):
            prd = tf.nn.softmax(output_)
            # (b * s, out) => (b, s, out)
            self.prd = tf.reshape(prd, [-1, time_step, output_size])

    def train(self, user_limit, start_learning_rate, training_steps, decay_rate):
        # data set
        train_x, train_y = DataSet(user_limit, self.time_step).lstm_train()

        # error and optimize function
        with tf.name_scope('train'):
            error = tf.reduce_mean(tf.abs(tf.subtract(self.prd, self.y)))
            tf.summary.scalar('error', error)
            # Dynamic learning rate
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, training_steps, decay_rate)
            tf.summary.scalar('learning_rate', learning_rate)
            update_op = tf.train.AdamOptimizer(learning_rate).minimize(error)

        # Run session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Merge summaries
            merged = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('saved_models/LSTM_%d/log' % self.hidden_unit, sess.graph)

            # initialize
            sess.run(tf.global_variables_initializer())

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
            saver = tf.train.Saver()
            saver.save(sess, 'saved_models/LSTM_%d/model' % self.hidden_unit, global_step=training_steps)

    def evaluate(self, user_limit):
        test_x, test_y, known = DataSet(user_limit, self.time_step).lstm_test()

        hits = 0
        crrs = 0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            module_file = tf.train.latest_checkpoint('saved_models/LSTM_%d/' % self.hidden_unit)
            saver = tf.train.Saver()
            saver.restore(sess, module_file)

            for i in range(len(test_x)):
                next_ = sess.run(self.prd, feed_dict={self.x: [test_x[i]]})

                priority = [[i, p] for p, i in enumerate(next_[0][-1])]
                priority.sort(reverse=True)
                priority = list(map(list, zip(*priority)))[1]

                rec = [g for g in priority if g not in known[i]][:50]
                crr = [h for h in test_y[i] if h not in known[i]]

                hits += len(set(rec).intersection(set(crr)))
                crrs += len(crr)

                print('[hits, corrs] = [%d, %d]' % (hits, crrs))

        print('recall = %g' % (hits / crrs))


if __name__ == '__main__':
    # network = LSTM(input_size=7649, hidden_unit=256, output_size=7649, time_step=8, batch_size=128)
    # network.train(user_limit=2000, start_learning_rate=0.001, training_steps=500, decay_rate=0.05)

    network = LSTM(input_size=7649, hidden_unit=256, output_size=7649, time_step=8)
    network.evaluate(user_limit=2000)
