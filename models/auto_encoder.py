import tensorflow as tf
from dataset import DataSet


class AutoEncoder:

    def __init__(self, input_size: int, hidden_unit: int, output_size: int):
        self.hidden_unit = hidden_unit

        self.x = tf.placeholder(tf.float32, shape=[None, input_size])
        self.y = tf.placeholder(tf.float32, shape=[None, output_size])

        with tf.name_scope('encode_layer'):
            with tf.name_scope('weights'):
                en_w = tf.Variable(tf.random_normal([input_size, hidden_unit]), name='encode_weights')
                tf.summary.histogram('weights', en_w)
            with tf.name_scope('biases'):
                en_b = tf.Variable(tf.constant(0.1, shape=[hidden_unit, ]), name='encode_biases')
                tf.summary.histogram('biases', en_b)
            hidden_layer = tf.nn.relu_layer(self.x, en_w, en_b)
        with tf.name_scope('decode_layer'):
            with tf.name_scope('weights'):
                de_w = tf.Variable(tf.random_normal([hidden_unit, output_size]), name='decode_weights')
                tf.summary.histogram('weights', de_w)
            with tf.name_scope('biases'):
                de_b = tf.Variable(tf.constant(0.1, shape=[output_size, ]), name='decode_biases')
                tf.summary.histogram('biases', de_b)
            self.output = tf.nn.relu_layer(hidden_layer, de_w, de_b)

    def train(self, user_limit, start_learning_rate, training_steps, decay_rate):
        # data set
        train = DataSet(user_limit).all_data()

        # error and optimize function
        with tf.name_scope('train'):
            error = tf.reduce_mean(tf.abs(self.output - self.y))
            tf.summary.scalar('error', error)
            # Dynamic learning rate
            global_step = tf.placeholder(tf.int16)
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, training_steps, decay_rate)
            tf.summary.scalar('learning_rate', learning_rate)
            update_op = tf.train.AdamOptimizer(learning_rate).minimize(error)

            # Run session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                # Merge summaries
                merged = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter('saved_models/AutoEncoder_%d/log' % self.hidden_unit, sess.graph)

                # initialize
                sess.run(tf.global_variables_initializer())

                # Start learning
                for turn in range(training_steps):
                    feed_dict = {}
                    err_sum = 0
                    curr_lr = 0

                    for input_ in train:
                        feed_dict = {self.x: input_, self.y: input_, global_step: turn}
                        _, curr_err, curr_lr = sess.run([update_op, error, learning_rate], feed_dict=feed_dict)
                        err_sum += curr_err

                    # Write summaries
                    summary = sess.run(merged, feed_dict=feed_dict)
                    summary_writer.add_summary(summary, global_step=turn)

                    print('Step %d: error = %g, learning rate = %g' % (turn, err_sum / len(train), curr_lr))

                # Save model
                saver = tf.train.Saver()
                saver.save(sess, 'saved_models/AutoEncoder_%d/model' % self.hidden_unit, global_step=training_steps)


if __name__ == '__main__':
    model = AutoEncoder(input_size=7649, hidden_unit=128, output_size=7649)
    model.train(user_limit=2000, start_learning_rate=0.001, decay_rate=0.3, training_steps=100)
