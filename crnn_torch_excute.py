import rnn_utils as input_reader
import data_utils as utils
import CRNN_torch as crnn
import Conf
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
import time 
import math

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def convert_2_Variable(item):
    return Variable(torch.from_numpy(np.array(h)))

criterion = nn.CrossEntropyLoss()

def train():
    config = Conf.SmallConfig()
    model = crnn.CRNN(True,config)
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden()
    train_input = input_reader.input_Producer(config,train_path='data/ptb.train.txt', word_path=utils.WORD_VOCAB_NAME, char_path=utils.CHAR_VOCAB_NAME, idx2wd_path=utils.IDX2WD_NAME)
    for step,(batch,target) in enumerate(train_input.producer(5)):
        char_batch, word_batch = [], []
        for batch_item in batch:
            char_batch.append(batch_item[0])
            word_batch.append(batch_item[2])
        char_ids, word_ids = Variable(torch.from_numpy(np.array(char_batch,dtype=np.int64))), Variable(torch.from_numpy(np.array(word_batch,dtype=np.int64)))
        target = Variable(torch.from_numpy(np.array(target,dtype=np.int64)).view(-1))
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(char_ids,word_ids,hidden)
        # print(output.view(-1,config.word_vocab_size).size())
        # print(target.size())
        loss = criterion(output.view(-1,config.word_vocab_size),target)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),config.max_grad_norm)
        for p in model.parameters():
            p.data.add_(-0.1, p.grad.data)

        total_loss += loss.data
        if step % 100 == 0 and step > 0:
            cur_loss = total_loss[0] / 100
            elapsed = time.time() - start_time
            print('|batch/ms {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(200/elapsed*1000, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

train()