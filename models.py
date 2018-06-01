import tensorflow as tf


class Models:
    @staticmethod
    def lstm(input_size: int, hidden_unit: int, output_size: int, time_step: int, batch_size: int):
        with tf.name_scope('input'):
            input_x = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
        with tf.name_scope('input layer'):
            with tf.name_scope('weights'):
                w_in = tf.Variable(tf.random_normal([input_size, hidden_unit]))
                tf.summary.histogram('weights', w_in)
            with tf.name_scope('biases'):
                b_in = tf.Variable(tf.constant(0.1, shape=[hidden_unit, ]))
                tf.summary.histogram('biases', b_in)
            with tf.name_scope('full connection'):
                # Reshape tensor to 2-dimension for matrix multiplication
                # (b, s, in) => (b * s, in)
                input_x = tf.reshape(input_x, [-1, input_size])
                # linear map
                # (b * s, in) => (b * s, h)
                input_rnn = tf.matmul(input_x, w_in) + b_in
                # Reshape tensor back to 3-dimension
                # (b * s, h) => (b, s, h)
                input_rnn = tf.reshape(input_rnn, [-1, time_step, hidden_unit])
        with tf.name_scope('lstm units'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_unit)
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            # output_rnn: results of every LSTM cells
            # final_states: states in the last LSTM cell
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
        with tf.name_scope('output layer'):
            with tf.name_scope('weights'):
                w_out = tf.Variable(tf.random_normal([hidden_unit, output_size]))
                tf.summary.histogram('weights', w_out)
            with tf.name_scope('biases'):
                b_out = tf.Variable(tf.constant(0.1, shape=[output_size, ]))
                tf.summary.histogram('biases', b_out)
            with tf.name_scope('full connection'):
                # (b, s, h) => (b * s, h)
                output_rnn = tf.reshape(output_rnn, [-1, hidden_unit])
                # (b * s, h) => (b * s, out)
                output_y = tf.matmul(output_rnn, w_out) + b_out
        return output_y, final_states
