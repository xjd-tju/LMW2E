import data_utils as utils
import itertools
import os
import sys
from smart_open import smart_open
import Conf

if sys.version_info[0] >= 3:
    unicode = str

def _to_unicode(text, encoding='utf8', errors='strict'):
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)

class RNNLineSentence(object):
    def __init__(self, source, length_limit, limit=None):
        self.source = source
        self.length_limit = length_limit
        self.limit = limit

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            self.source.seek(0)
            for line in itertools.islice(self.source, self.limit):
                line = _to_unicode(line.lower()).split()
                i = 0
                if len(line) > self.length_limit:
                    while (i + self.length_limit) <= len(line):
                        yield line[i: i + self.length_limit]
                        i += self.length_limit
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            if os.path.isdir(self.source):
                fileList = os.listdir(self.source)
                for file in fileList:
                    with smart_open(os.path.join(self.source, file)) as fin:
                        for line in itertools.islice(fin, self.limit):
                            line = _to_unicode(line.lower()).split()
                            i = 0
                            if len(line) > self.length_limit:
                                while (i + self.length_limit) <= len(line):
                                    yield line[i: i + self.length_limit]
                                    i += self.length_limit
            else:
                with smart_open(self.source) as fin:
                    for line in itertools.islice(fin, self.limit):
                        line = _to_unicode(line.lower()).split()
                        i = 0
                        if len(line) > self.length_limit:
                            while (i + self.length_limit) <= len(line):
                                yield line[i: i + self.length_limit]
                                i += self.length_limit

class input_Producer():
    def __init__(self, config,train_path, word_path, char_path, idx2wd_path):
        self.train_path = train_path
        self.word_vocab = utils.pkl_load(word_path)
        self.char_vocab = utils.pkl_load(char_path)
        self.idx2word = utils.pkl_load(idx2wd_path)
        self.iter_num = config.iter_num
        self.batch_size = config.batch_size
        self.time_steps = config.time_steps

    def producer(self,window_size):
        """
        Return:
        targets: [batch_size, time_steps]
        batch: [batch_size, time_steps]
        """
        if self.iter_num > 1:
            sentences = utils.RepeatCorpusNTimes(RNNLineSentence(self.train_path,self.time_steps+1), self.iter_num)
        else:
            sentences = RNNLineSentence(self.train_path, self.time_steps+1)
        # sentences = RNNLineSentence(self.train_path, self.time_steps)
        max_char_idx = len(self.char_vocab)
        batches, targets = [], []
        min_word_length = max(Conf.SmallConfig.char_kernel)
        for sentence in sentences:
            sentences_ids = [self.word_vocab[w].index for w in sentence if w in self.word_vocab]
            if len(sentences_ids) == self.time_steps+1:
                char_ids, chars_length, word_ids, words_length = self.batch_generate(window_size, sentences_ids, min_word_length)
                batches.append([char_ids, chars_length, word_ids, words_length])
                targets.append(sentences_ids[1:])
            if len(batches) == self.batch_size :
                batches = self.re_padding_char_ids(batches,max_char_idx)
                yield batches,targets  # each batches contains 20 sentences with variant length
                batches, targets = [], []

    def re_padding_char_ids(self,batches,max_char_idx):
        max_char_length = 0
        new_batch = []
        for batch_item in batches:
            char_ids_sequence, chars_length, word_ids, words_length = batch_item[0], batch_item[1], batch_item[2], batch_item[3]
            if max_char_length < max(chars_length):
                max_char_length = max(chars_length)
        for batch_item in batches:
            char_ids_sequence, chars_length, word_ids, words_length = batch_item[0], batch_item[1], batch_item[2], batch_item[3]
            new_char_ids_sequence = []
            for char_ids in char_ids_sequence:
                if len(char_ids) < max_char_length:
                    char_ids.extend([max_char_idx for _ in range(max_char_length-len(char_ids))])
                new_char_ids_sequence.append(char_ids)
            new_batch.append([new_char_ids_sequence, batch_item[1], batch_item[2], batch_item[3]])

        return new_batch

    def batch_generate(self, window_size, sentences_ids, min_word_length):
        """
        generate a batch for a sentence (a sequence of words)
        """
        max_word_idx, max_char_idx = len(self.word_vocab), len(self.char_vocab)
        char_ids_sequence, word_ids_sequence = [], []
        chars_length, words_length = [], []
        for i in range(len(sentences_ids)-1): #max value of i is len(word_ids)-2

            # if we write like this :
            # pos1, pos2 = max(i - window_size, 0), min(i + window_size, len(word_ids))
            # win_words_ids = word_ids[pos1:pos2]
            #when pos2 = len(word_ids), win_words_ids wouldn't out of range
            #when pos2 - i = 1, the last word of win_words_ids is target_word
            pos1, pos2 = max(i - window_size, 0), min(i + window_size, len(sentences_ids) - 1)
            target_word_id = sentences_ids[i]
            if (i-pos1) < window_size:
                pre_win_words = [max_word_idx for _ in range(window_size-i+pos1)]
                pre_win_words.extend(sentences_ids[pos1:i])
            else:
                pre_win_words = sentences_ids[pos1:i]
            if (pos2-i) < window_size:
                suf_win_words = sentences_ids[i:pos2+1]
                suf_win_words.extend([max_word_idx for _ in range(window_size-pos2+i)])
            else:
                suf_win_words = sentences_ids[i:pos2+1]
            win_words_ids = pre_win_words + suf_win_words
            char_ids = [self.char_vocab[c] for c in list(self.idx2word[target_word_id])]
            char_ids_sequence.append(char_ids)
            word_ids_sequence.append(win_words_ids)
            chars_length.append(len(char_ids))
            words_length.append(len(win_words_ids))
            # char_ids_sequence = self.re_padding_char_ids(char_ids_sequence,max(chars_length), max_char_idx)
        return  char_ids_sequence, chars_length, word_ids_sequence, words_length


