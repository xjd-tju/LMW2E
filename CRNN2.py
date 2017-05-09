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
        self.min_word_length = max(config.char_kernel)
        self.final_emb_size = len(config.char_kernel)+len(config.word_kernel)
        self.train(True)
        # self.cnn_model()
        # self.rnn_model(is_training,self.final_emb_size)

    def get_scope_variable(self,scope_name, var, shape, type):
        with tf.variable_scope(scope_name) as scope:
            try:
                v = tf.get_variable(var, shape, type)
            except ValueError:
                scope.reuse_variables()
                v = tf.get_variable(var)
        return v

    def conv2d(self,input, k_h, k_w, out_channel,name):
        # tf.nn.conv2d input size [batch, in_height, in_width, in_channels]
        # filter size [filter_height, filter_width, in_channels, out_channels]
        # output size [batch, height, width, channels]

        # different words have different length, so the batch size of CNN input must be 1
        # or use a max_word_length to truncate words whose length over the max_word_length
        w = self.get_scope_variable(name,'w',[k_h, k_w, 1, out_channel], tf.float32)
        b = self.get_scope_variable(name,'b',[out_channel], tf.float32)
        # with tf.variable_scope(name):
        #     w = tf.get_variable('w', [k_h, k_w, 1, out_channel])
        #     b = tf.get_variable('b', [out_channel])
        # amazing! there add a bias after convolution, and return the activation of the result
        return tf.tanh(tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='VALID') + b)

    def get_embedding(self, _input, padding, type):
        char_embedding = self.get_scope_variable('embedding',"char_embedding",[self.char_vocab_size, self.char_emb_size], tf.float32)
        word_embedding = self.get_scope_variable('embedding',"word_embedding",[self.word_vocab_size, self.word_emb_size], tf.float32)
        if type=="char":
            tmp_emb = tf.nn.embedding_lookup(char_embedding, _input)
        else:
            tmp_emb = tf.nn.embedding_lookup(word_embedding, _input)
        if padding is None:
            return tmp_emb
        else:
            zeros = tf.zeros([padding,self.char_emb_size], tf.float32)
            return tf.concat([tmp_emb,zeros],axis=0)

    def cnn_model(self,char_emb,word_emb):
        """
        compute a new emb only for one word each time
        """
        char_emb = tf.reshape(char_emb, shape=[1, -1, self.char_emb_size, 1])
        word_emb = tf.reshape(word_emb, shape=[1, -1, self.word_emb_size, 1])

        with tf.variable_scope('Conv'):
            char_pools, word_pools = [], []
            for k_h in self.char_kernel:
                char_conv = self.conv2d(char_emb, k_h, self.char_emb_size, 1, name="char_kernel_"+str(k_h))
                char_pool = tf.reduce_max(char_conv)
                char_pools.append(char_pool)
            for k_h in self.word_kernel:
                word_conv = self.conv2d(word_emb, k_h, self.word_emb_size, 1, name="word_kernel_"+str(k_h))
                word_pool = tf.reduce_max(word_conv)
                word_pools.append(word_pool)
        return tf.reshape(tf.concat([char_pools, word_pools], axis=0), [self.final_emb_size])

    # def get_new_emb(self,session,char_input,word_input):
    #     return session.run(self.final_emb,feed_dict={self.char_input:char_input,self.word_input:word_input})

    def rnn_model(self, _input, is_training):

        self._input = tf.reshape(_input, shape=[self.batch_size,self.time_steps, self.final_emb_size])
        def lstm_cell():
            # num units defined in the cell means the output size
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and self.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell( [attn_cell() for _ in range(self.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(self.batch_size, tf.float32)
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

    def train(self,is_training):
        self.char_batch = tf.placeholder(dtype=tf.int32,shape=[self.batch_size, self.time_steps,None])
        self.char_length = tf.placeholder(dtype=tf.int32,shape=[self.batch_size, self.time_steps])
        self.word_batch = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.time_steps, None])
        self.word_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.time_steps])
        new_batch = []
        for batch_idx in range(self.batch_size):
            new_sequence = []
            char_masks2D = tf.sequence_mask(self.char_length[batch_idx],maxlen=tf.reduce_max(self.char_length[batch_idx]))
            word_masks2D = tf.sequence_mask(self.word_length[batch_idx], maxlen=tf.reduce_max(self.word_length[batch_idx]))
            for time_idx in range(self.time_steps):
                char_ids = tf.boolean_mask(self.char_batch[batch_idx][time_idx],char_masks2D[time_idx])
                word_ids = tf.boolean_mask(self.word_batch[batch_idx][time_idx], word_masks2D[time_idx])
                char_emb = tf.cond(
                    pred=tf.less(self.char_length[batch_idx][time_idx], self.min_word_length),
                    fn1=lambda: self.get_embedding(char_ids, self.min_word_length - self.char_length[batch_idx][time_idx], type="char"),
                    fn2=lambda: self.get_embedding(char_ids, None, type="char")
                )
                word_emb = self.get_embedding(word_ids, None, type="word")
                new_sequence.append(self.cnn_model(char_emb,word_emb))
            new_batch.append(new_sequence)
        self.rnn_model(new_batch,is_training=is_training)





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
        char_ids, chars_length, word_ids, words_length = batch[0], batch[1], batch[2], batch[3]
        feed_dict = {model.char_batch: char_ids, model.char_length:chars_length, model.word_batch:word_ids, model.word_length: words_length, model._target: target}
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]

        costs += cost
        iters += model.time_steps

        if verbose and step % 100 == 1:
            print("%d perplexity: %.3f speed: %.2f b/s" % (step, np.exp(costs / iters), model.batch_size / (time.time() - start_time)))
            start_time = time.time()

    return np.exp(costs / iters)




