import numpy as np
import tensorflow as tf

# Constant
hidden_unit = 8  # hidden layer units
input_size = 5
output_size = 5
lr = 0.0005  # Learning rate

# LSTM parameters
time_step = 10
batch_size = 20
X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])

# Data
data = [[0.5 for a in range(b, b + 5)] for b in range(101)]

train_x, train_y = [], []
for i in range(len(data) - time_step - 1):
    x = data[i:i + time_step]
    y = data[i + 1:i + time_step + 1]
    train_x.append(x)
    train_y.append(y)


def lstm():
    # Define Input weights and biases
    w_in = tf.Variable(tf.random_normal([input_size, hidden_unit]))
    b_in = tf.Variable(tf.constant(0.1, shape=[hidden_unit, ]))
    # Change tensor to 2-dimension
    # Results are input to hidden layer
    input_ = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input_, w_in) + b_in
    # Change tensor to 3-dimension
    # Results are input to to LSTM cells
    input_rnn = tf.reshape(input_rnn, [-1, time_step, hidden_unit])
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # output_rnn: results of every LSTM cells
    # final_states: result of the last LSTM cell
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)
    # Input to output layer
    output = tf.reshape(output_rnn, [-1, hidden_unit])
    w_out = tf.Variable(tf.random_normal([hidden_unit, output_size]))
    b_out = tf.Variable(tf.constant(0.1, shape=[output_size, ]))
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


def train():
    pred, _ = lstm()
    # loss function
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    # saver = tf.train.Saver(tf.global_variables())
    # module_file = tf.train.latest_checkpoint()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        # Training times
        for i in range(1000):
            start = 0
            end = batch_size
            while end < len(train_x):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start:end]})
                start += batch_size
                end += batch_size
                print(i, loss_)


if __name__ == '__main__':
    train()
