import numpy as np
import rnn_utils as input_reader
from data_utils import Text_vocab_builder
import data_utils as utils
import Conf
import torch

def main():
    config = Conf.SmallConfig()
    # Text_vocab_builder(
    #     max_vocab_size=config.word_vocab_size,
    #     min_word_count=1,
    #     sub_sample=None,
    #     vocab_path='',
    #     sentences=utils.LineSentence('data\\ptb.train.txt')
    # )


    ip = input_reader.input_Producer(config,train_path='data/ptb.train.txt', word_path=utils.WORD_VOCAB_NAME, char_path=utils.CHAR_VOCAB_NAME, idx2wd_path=utils.IDX2WD_NAME)
    for batch,t in ip.producer(5):
        print("new_batches")
        char_batch, word_batch = [], []
        for batch_item in batch:
            # char_batch.append(batch_item[0])
            # word_batch.append(batch_item[2])
            for item in batch_item[2]:
                print(len(item))
        # char_np = np.array(word_batch,dtype=np.int32)
        # a = torch.from_numpy(np.array(word_batch,dtype=np.int32))
        # print(a.size())
    # with tf.Graph().as_default():
    #     initializer = tf.random_uniform_initializer(-1,1)
    #     with tf.name_scope("Train"):
    #         train_input = input_reader.input_Producer(config,train_path='data\\ptb.train.txt', word_path=utils.WORD_VOCAB_NAME, char_path=utils.CHAR_VOCAB_NAME, idx2wd_path=utils.IDX2WD_NAME)
    #         with tf.variable_scope("Model", reuse=None, initializer=initializer):
    #             m = crnn.CRNNModel(True, config)
    #         tf.summary.scalar("Training Loss", m._cost)
    #         tf.summary.scalar("Learning Rate", m._lr)

        # with tf.Session() as session:
        #     session.run(tf.global_variables_initializer())
        #     for i in range(config.iter_num):
        #         train_perplexity = crnn.run_epoch(session, m, inputs=train_input, win_size=max(config.word_kernel), train_op=m._train_op, verbose=True)
        #         print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

main()