import tensorflow as tf


class Models:
    @staticmethod
    def lstm(input_: tf.Tensor, input_size: int, hidden_unit: int, output_size: int, time_step: int, batch_size: int):
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
                input_ = tf.reshape(input_, [-1, input_size])
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
            prd = tf.reshape(prd, [-1, time_step, output_size])
        return prd

    @staticmethod
    def codec(input_: tf.Tensor, input_size: int, hidden_unit: int):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([input_size, hidden_unit]))
            tf.summary.histogram('weights', weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.1, shape=[hidden_unit, ]))
            tf.summary.histogram('biases', biases)
        with tf.name_scope('encode_layer'):
            hidden_cells = tf.matmul(input_, weights) + biases
        with tf.name_scope('decode_layer'):
            output_ = tf.matmul(hidden_cells - biases, tf.transpose(weights))
        return hidden_cells, output_
