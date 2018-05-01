"""Primary app to carry out analysis on word frequency model."""
import logging

import tensorflow as tf


class Model():
    """The TensorFlow model specification for this idea."""

    def __init__(self, args, batch_size, mode='train'):
        """The standard __init__ function."""
        logger = logging.getLogger(__name__)

        self.args = args
        self.config = config = self.args.config

        # Defining the epoch variables
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_incr = self.epoch.assign(self.epoch + 1)
        self.global_step = tf.Variable(0, trainable=False)

        # Used to update training schedule
        self.best_ppl = tf.Variable(10000.0, trainable=False, dtype=tf.float32)
        self.best_ppl_new = tf.placeholder(tf.float32, shape=())
        self.best_ppl_assign = self.best_ppl.assign(self.best_ppl_new)

        self.margin_ppl = tf.Variable(10000.0, trainable=False, dtype=tf.float32)
        self.margin_ppl_new = tf.placeholder(tf.float32, shape=())
        self.margin_ppl_assign = self.margin_ppl.assign(self.margin_ppl_new)

        self.last_ppl_update = tf.Variable(0, trainable=False)
        self.last_ppl_update_new = tf.placeholder(tf.int32, shape=())
        self.last_ppl_update_assign = self.last_ppl_update.assign(self.last_ppl_update_new)

        # Defining the loss interpolation constant
        self.l1 = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.l1_new = tf.placeholder(tf.float32, shape=())
        self.l1_assign = self.l1.assign(self.l1_new)
        self.l2 = 1.0 - self.l1

        self.input_data = tf.placeholder(tf.int32, [batch_size, config.timesteps])
        self.targets = tf.placeholder(tf.int32, [batch_size, config.timesteps])
        # These placeholders expect a log-probability distribution on axis=2
        self.ngram = tf.placeholder(tf.float32, [batch_size, config.timesteps, args.vocab_size])

        # Taking inputs, applying dropout, passing through embeddings
        self.embedding = embedding = tf.get_variable("embedding", [args.vocab_size, config.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        if mode == 'train':
            inputs = tf.nn.dropout(inputs, keep_prob=config.input_keep_prob)

        # The whole BasicLSTMCell network
        cells = []
        initial_states = []
        for i in range(config.num_layers):
            cell = tf.nn.rnn_cell.BasicLSTMCell(
                config.rnn_size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse
            )
            if mode == 'train':
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell=cell,
                    output_keep_prob=config.intra_keep_prob,
                    state_keep_prob=config.state_keep_prob,
                    variational_recurrent=True,
                    dtype=tf.float32
                )
            cells.append(cell)
            initial_states.append(cell.zero_state(batch_size, tf.float32))
        self.cells = tuple(cells)
        self.initial_states = tuple(initial_states)

        # The actual LSTM computation, `self.initial_state` will be fed later on
        final_states = []
        outputs = []
        for i in range(config.num_layers):
            with tf.variable_scope("layer%d" % i):
                inputs, final_state = tf.nn.dynamic_rnn(
                    self.cells[i], inputs, initial_state=self.initial_states[i]
                )
            outputs.append(inputs)
            final_states.append(final_state)
        self.final_states = tuple(final_states)
        # Skip connections to make training easier
        self.outputs = tf.add_n(outputs)

        with tf.variable_scope('logits'):
            # Layer of logits before softmax after RNN
            if config.shared_embeddings is True:
                self.softmax_w = softmax_w = tf.transpose(embedding, [1, 0])
            else:
                self.softmax_w = softmax_w = tf.get_variable("softmax_w", [config.rnn_size, args.vocab_size])
            self.softmax_b = softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        # The output dropout has been applied in the DropoutWrapper
        output = tf.reshape(self.outputs, [-1, config.rnn_size])
        self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # Store the actual probability values.
        # Used by evaluation function in some cases
        self.probs = tf.nn.softmax(self.logits)

        # Converting the distribution to a one hot vector
        self.distro1 = tf.reshape(tf.one_hot(self.targets, args.vocab_size), [-1, args.vocab_size])
        # Finding 1-D cross entropy loss tensor
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.distro1, logits=self.logits)
        # Scaling by interpolation values of L1
        self.cost = tf.reduce_sum(self.loss) / batch_size

        # This is already a valid distribution. Just needs reshaping to 2D form
        ngram_exp = tf.exp(self.ngram)
        self.distro2 = tf.reshape(ngram_exp, [-1, args.vocab_size])
        # Finding 1-D cross entropy loss tensor
        self.loss2 = tf.nn.softmax_cross_entropy_with_logits(labels=self.distro2, logits=self.logits)
        # Scaling by interpolation values of L2
        self.cost2 = tf.reduce_sum(self.loss2) / batch_size

        # self.final_cost = self.cost

        self.final_cost = tf.add(self.l1 * self.cost, self.l2 * self.cost2)

        if mode == 'eval':
            return

        # Defining the learning rate variables
        self.lr = tf.Variable(config.lr, trainable=False)
        self.lr_decay = self.lr.assign(self.lr * config.lr_decay)

        # Standard tricks to train LSTMs
        tvars = tf.trainable_variables()
        for variable in tvars:
            logger.info("%s - %s", variable.name, str(variable.get_shape()))
        self.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.final_cost, tvars),
            config.grad_clip
        )

        if config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(
            zip(self.grads, tvars),
            global_step=self.global_step
        )

        # Model savers
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
