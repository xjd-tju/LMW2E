import time

import numpy as np
import tensorflow as tf


class CRNNModel(object):
    def __init__(self, is_training, config):
        self.batch_size = config.batch_size
        self.time_steps = config.time_steps
        self.hidden_size = config.hidden_size
        self.word_vocab_size = config.word_vocab_size
        self.char_vocab_size = config.char_vocab_size
        self.keep_prob = config.keep_prob
        self.num_layers = config.num_layers
        self.max_grad_norm = config.max_grad_norm
        self.char_emb_size = config.char_emb_size
        self.word_emb_size = config.word_emb_size
        self.char_kernel = config.char_kernel
        self.word_kernel = config.word_kernel
        self.final_emb_size = len(self.char_kernel)+len(self.word_kernel)
        self.cnn_model()
        self.rnn_model(is_training,self.final_emb_size)

    def conv2d(self,input, k_h, k_w, out_channel,name):
        # tf.nn.conv2d input size [batch, in_height, in_width, in_channels]
        # filter size [filter_height, filter_width, in_channels, out_channels]
        # output size [batch, height, width, channels]

        # different words have different length, so the batch size of CNN input must be 1
        # or use a max_word_length to truncate words whose length over the max_word_length
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, 1, out_channel])
            b = tf.get_variable('b', [out_channel])
        # amazing! there add a bias after convolution, and return the activation of the result
        return tf.tanh(tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='VALID') + b)

    def cnn_model(self):
        """
        compute a new emb only for one word each time
        """
        self.char_input = tf.placeholder(dtype=tf.int32, shape=[None])
        self.word_input = tf.placeholder(dtype=tf.int32, shape=[None])
        self.char_embedding = tf.get_variable("char_embedding", [self.char_vocab_size, self.char_emb_size], tf.float32)
        self.word_embedding = tf.get_variable("word_embedding", [self.word_vocab_size, self.word_emb_size], tf.float32)
        char_emb = tf.nn.embedding_lookup(self.char_embedding, self.char_input)
        word_emb = tf.nn.embedding_lookup(self.word_embedding, self.word_input)
        self.char_emb = tf.reshape(char_emb, shape=[1, -1, self.char_emb_size, 1])
        self.word_emb = tf.reshape(word_emb, shape=[1, -1, self.word_emb_size, 1])

        with tf.variable_scope('Conv'):
            char_pools, word_pools = [], []
            for k_h in self.char_kernel:
                # reduced_length = tf.shape(self.char_emb)[1] - k_h + 1
                char_conv = self.conv2d(self.char_emb, k_h, self.char_emb_size, 1, name="char_kernel_"+str(k_h))
                # char_pool = tf.nn.max_pool(char_conv, tf.shape(char_conv), [1, 1, 1, 1], 'VALID')
                char_pool = tf.reduce_max(char_conv)
                char_pools.append(char_pool)
            for k_h in self.word_kernel:
                # reduced_length = self.word_emb.get_shape().as_list()[1] - k_h + 1
                word_conv = self.conv2d(self.word_emb, k_h, self.word_emb_size, 1, name="word_kernel_"+str(k_h))
                # word_pool = tf.nn.max_pool(word_conv, tf.shape(word_conv), [1, 1, 1, 1], 'VALID')
                word_pool = tf.reduce_max(word_conv)
                word_pools.append(word_pool)
        # char_pools = tf.concat(char_pools, axis=0)
        # word_pools = tf.concat(word_pools, axis=0)
        self.final_emb = tf.reshape(tf.concat([char_pools, word_pools], axis=0), [self.final_emb_size])

    def get_new_emb(self,session,char_input,word_input):
        return session.run(self.final_emb,feed_dict={self.char_input:char_input,self.word_input:word_input})

    def rnn_model(self, _input, is_training, final_emb_size):

        def lstm_cell():
            # num units defined in the cell means the output size
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and self.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell( [attn_cell() for _ in range(self.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(self.batch_size, tf.float32)
        # self._input = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.time_steps, final_emb_size])
        self._input = _input
        self._target = tf.placeholder(dtype=tf.int32,shape=[self.batch_size, self.time_steps])

        # if is_training and self.keep_prob < 1:
        #     inputs = tf.nn.dropout(self._input, self.keep_prob)

        state = self._initial_state
        #
        # inputs with with time major is false : [batch_size, time_steps, embedding_size]
        # output size : [batch_size, time_steps,output_size]
        #
        outputs, state = tf.nn.dynamic_rnn(cell, self._input, initial_state=state)
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.hidden_size])
        softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.word_vocab_size], tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.word_vocab_size], tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._target, [-1])],
            [tf.ones([self.batch_size * self.time_steps], tf.float32)])
        self._cost = tf.reduce_sum(loss) / self.batch_size
        self._final_state = state

        if not is_training:
            return
        # if not train model, we don't need to update params and update learning rate
        self._lr = tf.Variable(1.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),self.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients( zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        return session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


def run_epoch(session, model, time_steps, inputs, win_size, train_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0

    fetches = {
        "cost": model._cost,
        "final_state": model._final_state,
    }
    if train_op is not None:
        fetches["train_op"] = train_op
    for step,(batch,target) in enumerate(inputs.producer(win_size)):
        new_batch = []
        for item in batch:
            new_sequence = []
            for i in range(time_steps):
                new_sequence.append(model.get_new_emb(session,item[0][i],item[1][i]))
            new_batch.append(new_sequence)

        feed_dict = {model._input: new_batch, model._target: target}
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]

        costs += cost
        iters += model.time_steps

        if verbose and step % 100 == 1:
            print("%d perplexity: %.3f speed: %.2f b/s" % (step, np.exp(costs / iters), model.batch_size / (time.time() - start_time)))
            start_time = time.time()

    return np.exp(costs / iters)




