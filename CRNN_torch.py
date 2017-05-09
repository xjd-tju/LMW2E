import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class CRNN(nn.Module):
    """docstring for CRNN"""
    def __init__(self, is_training, config):
        super(CRNN, self).__init__()
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
        self.char_kernel_size = config.char_kernel
        self.word_kernel_size = config.word_kernel
        self.min_word_length = max(config.char_kernel)
        self.final_emb_size = len(config.char_kernel)+len(config.word_kernel)
        self.rnn_type = config.rnn_type

        self.word_embedding = nn.Embedding(config.word_vocab_size+1,config.word_emb_size,padding_idx = config.word_vocab_size)
        self.char_embedding = nn.Embedding(config.char_vocab_size+1,config.char_emb_size,padding_idx = config.char_vocab_size)
        self.char_kernel = {}
        # self.char_pooling = {}
        for k_h in self.char_kernel_size:
            self.char_kernel["ck_"+str(k_h)] = nn.Conv2d(1,1,(k_h,config.char_emb_size))
            # self.char_pooling["cp_"+str(k_h)] = nn.Conv2d(k_h,1)
        self.word_kernel = {}
        # self.word_pooling = {}
        for k_h in self.word_kernel_size:
            self.word_kernel["wk_"+str(k_h)] = nn.Conv2d(1,1,(k_h,config.word_emb_size))
            # self.word_pooling["wp_"+str(k_h)] = nn.Conv2d(k_h,config.word_emb_size)
        if config.rnn_type in ["LSTM","GRU"]:
            self.rnn = getattr(nn,config.rnn_type)(self.final_emb_size, config.hidden_size, config.num_layers, batch_first=True)
        else:
            try:
                nonlinearity = {'RNN_TANH':'tanh','RNN_RELU':'relu'}[rnn_type]
            except KeyError:
                raise ValueError("an invalid option for model was supplied, options are['LSTM','GRU','RNN_TANH','RNN_RELU']")
            self.rnn = nn.RNN(self.final_emb_size, config.hidden_size, config.num_layers,nonlinearity=nonlinearity, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, self.word_vocab_size)
        self.fc1.weight.data.uniform_(-1,1)
        self.fc1.bias.data.fill_(0)

    def forward(self,char_inputs,word_inputs,hidden):
        rnn_inputs_list = []
        for batch_idx in range(self.batch_size):# batch size : [batch_size, time_steps, max_char_length] , [batch_size, time_steps, win_word_num]
            char_embed = self.char_embedding(char_inputs[batch_idx]) # char_embed size: [sentence_length, max_char_length, char_embedding_size]
            char_embed = char_embed.view(char_embed.size()[0], 1, char_embed.size()[1], char_embed.size()[2]) # [sentence_length, channel, max_char_length, char_emb_size]
            word_embed = self.word_embedding(word_inputs[batch_idx]) # [sentence_length, window_words_size, word_embedding_size]
            word_embed = word_embed.view(word_embed.size()[0], 1, word_embed.size()[1], word_embed.size()[2]) # [sentence_length, channel, win_word_num, char_emb_size]
            char_conv_list, word_conv_list = [], []
            for k_h in self.char_kernel_size:
                tmp = F.max_pool2d(self.char_kernel["ck_"+str(k_h)](char_embed),(char_embed.size()[2]-k_h+1, 1))
                char_conv_list.append(tmp.view(self.time_steps,1))
            new_char_emb = torch.cat(tuple(char_conv_list),1)
            for k_h in self.word_kernel_size:
                tmp = F.max_pool2d(self.word_kernel["wk_"+str(k_h)](word_embed),(word_embed.size()[2]-k_h+1, 1))
                word_conv_list.append(tmp.view(self.time_steps,1)) # tmp size after squeeze is [sentence_length]
            new_word_emb = torch.cat(tuple(word_conv_list),1) # new_word_emb size is [sentence_length, kernel_num] 
            final_emb = torch.cat((new_char_emb,new_word_emb),1).view(1, char_embed.size()[0], self.final_emb_size) #[1, time_steps, final_emb_size]
            rnn_inputs_list.append(final_emb)
        rnn_inputs = torch.cat(tuple(rnn_inputs_list), 0) #size: [batch_size, time_steps, emb_size]
        # print(rnn_inputs.size())
        out, hidden = self.rnn(rnn_inputs, hidden)
        line_tmp = out.contiguous().view(out.size(0)*out.size(1),out.size(2))
        output = self.fc1(line_tmp)
        return output.view(out.size(0),out.size(1),output.size(1)), hidden

    def init_hidden(self):
        if self.rnn_type == "LSTM":
            return (Variable(torch.FloatTensor(self.num_layers, self.batch_size,self.hidden_size).zero_())
                ,Variable(torch.FloatTensor(self.num_layers, self.batch_size,self.hidden_size).zero_()))
        else:
            return Variable(torch.FloatTensor(self.num_layers, self.batch_size,self.hidden_size).zero_())

