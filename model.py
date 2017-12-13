import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.legacy_seq2seq as seq2seq


class Model:
    def __init__(self, data, model='lstm', infer=False):
        self.rnn_size = 256
        self.n_layers = 2

        if infer:
            self.batch_size = 1
        else:
            self.batch_size = data.batch_size

        if model == 'rnn':
            cell_rnn = rnn.BasicRNNCell
        elif model == 'gru':
            cell_rnn = rnn.GRUCell
        elif model == 'lstm':
            cell_rnn = rnn.BasicLSTMCell

        cell = cell_rnn(self.rnn_size, state_is_tuple=False)
        self.cell = rnn.MultiRNNCell([cell] * self.n_layers, state_is_tuple=False)

        self.x_tf = tf.placeholder(tf.int32, [self.batch_size, None])
        self.y_tf = tf.placeholder(tf.int32, [self.batch_size, None])

        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [self.rnn_size, data.words_size])
            softmax_b = tf.get_variable("softmax_b", [data.words_size])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                    "embedding", [data.words_size, self.rnn_size])
                inputs = tf.nn.embedding_lookup(embedding, self.x_tf)

        outputs, final_state = tf.nn.dynamic_rnn(
            self.cell, inputs, initial_state=self.initial_state, scope='rnnlm')

        self.output = tf.reshape(outputs, [-1, self.rnn_size])
        self.logits = tf.matmul(self.output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        self.final_state = final_state
        pred = tf.reshape(self.y_tf, [-1])
        # seq2seq
        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [pred],
                                                [tf.ones_like(pred, dtype=tf.float32)],)

        self.cost = tf.reduce_mean(loss)
        self.learning_rate = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
