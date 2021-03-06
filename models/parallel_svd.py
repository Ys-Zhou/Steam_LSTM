import tensorflow as tf
from dataset import DataSet


class PrlSVD:

    def __init__(self, item_size: int, user_size: int, hidden_size: int, time_step: int):
        self.user_size = user_size
        self.hidden_size = hidden_size

        self.y = tf.placeholder(tf.float32, shape=[user_size, time_step, item_size], name='train_data')
        self.sign = tf.placeholder(tf.float32, shape=[user_size, time_step, item_size], name='sign_data')

        with tf.name_scope('item_matrix'):
            item_mtx = tf.Variable(tf.random_normal([hidden_size, item_size]))
            self.saver = tf.train.Saver({'map_mtx': item_mtx})
        with tf.name_scope('user_matrix'):
            user_mtx = tf.Variable(tf.random_normal([time_step, user_size, hidden_size]))
        with tf.name_scope('like_svd'):
            user_mtx = tf.reshape(user_mtx, [-1, hidden_size])
            output = user_mtx @ item_mtx
            output = tf.reshape(output, [-1, user_size, item_size])
            self.output = tf.transpose(output, perm=[1, 0, 2])

    def train(self, start_learning_rate: float, training_steps: int, decay_rate: float):
        # data set
        train_data, train_sign = DataSet(self.user_size).all_data()

        # error and optimize function
        with tf.name_scope('train'):
            # Use sign data to ignore 0 in train data
            error = tf.reduce_mean(tf.abs(self.output * self.sign - self.y))
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
            summary_writer = tf.summary.FileWriter('saved_models/Prl_SVD_%d/log' % self.hidden_size, sess.graph)

            # initialize model
            sess.run(tf.global_variables_initializer())

            # Start learning
            for turn in range(training_steps):
                feed_dict = {self.y: train_data, self.sign: train_sign, global_step: turn}
                _, curr_err, curr_lr, summary = sess.run([update_op, error, learning_rate, merged], feed_dict=feed_dict)

                # Write summaries
                summary_writer.add_summary(summary, global_step=turn)

                print('Step %d: error = %g, learning rate = %g' % (turn, curr_err, curr_lr))

            # Save model
            self.saver.save(sess, 'saved_models/Prl_SVD_%d/model' % self.hidden_size, global_step=training_steps)


if __name__ == '__main__':
    model = PrlSVD(item_size=7649, user_size=2000, hidden_size=256, time_step=12)
    model.train(start_learning_rate=0.1, decay_rate=0.002, training_steps=1000)
