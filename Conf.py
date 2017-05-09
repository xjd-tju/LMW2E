class SmallConfig(object):
    """Small config."""
    iter_num = 10
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    time_steps = 20
    hidden_size = 50
    keep_prob = 0.9
    lr_decay = 0.5
    batch_size = 30
    word_vocab_size = 10000
    char_vocab_size = 47
    char_emb_size = 20
    word_emb_size = 100
    char_kernel = [1,2,3,4,5]
    word_kernel = [1,2,3,4,5]
    rnn_type = "LSTM" #['LSTM','GRU','RNN_TANH','RNN_RELU']")