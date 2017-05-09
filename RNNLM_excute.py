import tensorflow as tf
import rnn_utils as input_reader
from data_utils import Text_vocab_builder
import data_utils as utils
import CRNN2 as crnn
import Conf

def main(_):
    config = Conf.SmallConfig()
    Text_vocab_builder(
        max_vocab_size=config.word_vocab_size,
        min_word_count=1,
        sub_sample=None,
        vocab_path='',
        sentences=utils.LineSentence('data\\ptb.train.txt')
    )

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-1,1)
        with tf.name_scope("Train"):
            train_input = input_reader.input_Producer(config,train_path='data\\ptb.train.txt', word_path=utils.WORD_VOCAB_NAME, char_path=utils.CHAR_VOCAB_NAME, idx2wd_path=utils.IDX2WD_NAME)
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = crnn.CRNNModel(True, config)
            tf.summary.scalar("Training Loss", m._cost)
            tf.summary.scalar("Learning Rate", m._lr)

        # with tf.name_scope("Test"):
        #     test_input = input_reader.input_Producer(config, path='data\\ptb.test.txt', vocab_path='vocab')
        #     with tf.variable_scope("Model", reuse=True, initializer=initializer):
        #         mtest = rnn.RNNModel(False, config)

        # sv = tf.train.Supervisor(logdir='save')
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
        # with sv.managed_session() as session:
            for i in range(config.iter_num):
                # lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                # lr = m.assign_lr(session, config.learning_rate * lr_decay)
                #
                # print("Epoch: %d Learning rate: %.3f" % (i + 1, lr))
                train_perplexity = crnn.run_epoch(session, m, inputs=train_input, time_steps=config.time_steps, win_size=max(config.word_kernel), train_op=m._train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            # test_perplexity = rnn.run_epoch(session, mtest,inputs=test_input)
            # print("Test Perplexity: %.3f" % test_perplexity)

            # print("Saving model to %s." % 'save')
            # sv.saver.save(session, 'save', global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()