import time

import numpy as np
import tensorflow as tf


class RNNModel(object):
    def __init__(self, is_training, config):
        self.batch_size = config.batch_size
        self.time_steps = config.time_steps
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.keep_prob = config.keep_prob
        self.num_layers = config.num_layers
        self.max_grad_norm = config.max_grad_norm
        self.model(is_training)

    def model(self,is_training):

        def lstm_cell():
            # num units defined in the cell means the output size
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and self.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell( [attn_cell() for _ in range(self.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(self.batch_size, tf.float32)
        self._input = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.time_steps])
        self._target = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.time_steps])
        embedding = tf.get_variable( "embedding", [self.vocab_size, self.hidden_size], tf.float32)
        with tf.device("/cpu:0"):
            inputs = tf.nn.embedding_lookup(embedding, self._input)

        if is_training and self.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, self.keep_prob)

        state = self._initial_state
        #
        # inputs with with time major is false : [batch_size, time_steps, embedding_size]
        # output size : [batch_size, time_steps,output_size]
        #
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=state)
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.hidden_size])
        softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.vocab_size], tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size], tf.float32)
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

class SmallConfig(object):
    """Small config."""
    iter_num = 1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 1
    time_steps = 10
    hidden_size = 200
    keep_prob = 0.9
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

def run_epoch(session, model, inputs, train_op=None, verbose=False):
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

    for step,(batch,target) in enumerate(inputs.producer()):
        feed_dict = {model._input: batch, model._target: target}
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]

        costs += cost
        iters += model.time_steps

        if verbose and step % 100 == 1:
            print("%d perplexity: %.3f speed: %.2f b/s" % (step, np.exp(costs / iters), model.batch_size / (time.time() - start_time)))
            start_time = time.time()

    return np.exp(costs / iters)




