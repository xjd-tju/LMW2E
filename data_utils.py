# -*- coding: utf-8 -*-

import pickle
import sys
import itertools
import logging
import codecs
import numpy as np
import random
import os
import collections
import re
from smart_open import smart_open
from six import iteritems, itervalues
from types import GeneratorType

if sys.version_info[0] >= 3:
    unicode = str

logger = logging.getLogger("Utils:")

import time


# --exeTime
def execute_Time(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        print("function {%s} takes %.3fs" % (func.__name__, time.time() - t0))
        return back

    return newFunc


def vocab_frmEvl(test_path, num, vocab_path):
    logger = logging.getLogger('construct evl vocab:')
    test_vocab = {}
    text = []
    with open(test_path, 'r') as f:
        sentences = f.readlines()
        for sentence in sentences:
            words_list = sentence.lower().split()
            if len(words_list) >= num:
                text.extend(words_list[:num])

    vocab = pkl_load(vocab_path)
    for word in set(text):
        if word in vocab:
            test_vocab[word] = Vocab(count=0, index=vocab[word].index)

    logger.info('evl vocab contains %d unique words:', len(test_vocab))
    return test_vocab


def pick_sentences(train_path, question_path, vocab_path, sentence_save_path, evl_vocab_path):
    logger = logging.getLogger('pick sentences:')
    list_file = os.listdir(train_path)
    evl_vocab = vocab_frmEvl(question_path, 4, vocab_path)
    pattern = re.compile("^[a-z]+$", re.IGNORECASE)
    sentence_num = 0
    with codecs.open(sentence_save_path, 'w', encoding='utf-8') as output:
        for file in list_file:
            snum, pnum = 0, 0
            sentences = LineSentence(os.path.join(train_path, file))
            for sentence in sentences:
                snum += 1
                flag = False
                for word in sentence[:]:
                    if re.match(pattern, word):
                        sentence.remove(word)
                    if word in evl_vocab and flag is False:
                        flag = True

                if flag is True:
                    new_sentence = " ".join(sentence).strip()
                    output.write(new_sentence)
                    output.write('\n')
                    pnum += 1

                for word in sentence:
                    if word in evl_vocab:
                        evl_vocab[word].count += 1
            sentence_num += pnum
            logger.info('%s contains %d sentences, %d were picked up', file, snum, pnum)

    pkl_save(evl_vocab, evl_vocab_path)
    return sentence_num


def to_unicode(text, encoding='utf8', errors='strict'):
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


def pkl_save(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file, protocol=2)


def emb_save(file_name, embeddings):
    np.save(file_name, embeddings)


def pkl_load(path):
    with open(path, 'rb') as file:
        vocab = pickle.load(file, encoding='latin1')
    return vocab


def line_number(path):
    with open(path, 'r', encoding='utf-8') as f:
        line_num = sum(1 for line in f)
        return line_num


def dump2text(question_list, path, split_signal=","):
    # [1, 2, 3, 5] -->  1,2,3,5
    with open(path, 'w') as file:
        for item in question_list:
            file.write(split_signal.join(str(i) for i in item))
            file.write('\n')


def result2list(path):
    result = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            result.append(line.split(','))
    return result


def load_question(type, path, vocab=None):
    """
    Read questions from files and return a dictionary. the key is questions category, and the value is a list which contains questions.
    If we specify a vocab, the question word will be replaced by its index in vocab, like [1,2,3,5]
    else the question word is just a list of str, like ['one','two','three','four']
    """
    questions = {}
    total = 0
    skip = 0
    word_list = []
    if type == 1:
        with open(path, 'r') as eval_file:
            lines = eval_file.readlines()
            question_name = ''
            for line in lines:
                if line.startswith(':'):
                    question_name = line.strip()
                    questions[question_name] = []
                    continue
                question = line.strip().lower().split()
                if len(question) == 4:
                    word_list.extend(question)
                    if vocab is not None:
                        try:
                            ids = [vocab[item].index for item in question]
                            questions[question_name].append(np.array(ids))
                            total += 1
                        except:
                            skip += 1
                    else:
                        questions[question_name].append(question)
                        total += 1

                        # 将questions的值转换为array类型
            if vocab is not None:
                for name in questions.keys():
                    questions[name] = np.array(questions[name])

        word_list = set(word_list)
        print(
            'Finish reading analogy questions. %d kinds, containing %d unique words,%d questions are loaded, %d questions are skipped, '
            % (len(questions), len(word_list), total, skip))
        return total, questions


def sub_questions_dump(num, questions_path, sub_path, name_list=None):
    """
    if name_list is not None, we just extract questions whose name is in name_list
    else if name_list is None, we averagely subsample each kind of questions, the size of total sub_questions is num
    """
    total, qs = load_question(type=1, path=questions_path)
    num = num // len(qs)
    total = 0
    vocab = []
    if name_list is not None:
        with codecs.open(sub_path, 'w', encoding='utf-8') as output:
            for name in qs:
                if name in name_list:
                    output.write(name)
                    output.write('\n')
                    new_questions = qs[name]
                    for question in new_questions:
                        vocab.extend(question)
                        new_question = " ".join(question).strip()
                        output.write(new_question)
                        output.write('\n')

                    total += len(new_questions)
    else:
        with codecs.open(sub_path, 'w', encoding='utf-8') as output:
            for name in qs:
                if num < len(qs[name]):
                    new_questions = random.sample(qs[name], num)

                else:
                    new_questions = qs[name]
                for question in new_questions:
                    vocab.extend(question)
                    new_question = " ".join(question).strip()
                    output.write(new_question)
                    output.write('\n')

                total += len(new_questions)
    vocab = set(vocab)
    print('sub sampling %d questions, containing %d words' % (total, len(vocab)))


class RepeatCorpusNTimes():
    def __init__(self, corpus, n):
        self.corpus = corpus
        self.n = n

    def __iter__(self):
        for _ in range(self.n):
            for document in self.corpus:
                yield document


class Vocab(object):
    """
    A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).

    """

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class LineSentence(object):
    """
    Simple format: one sentence = one line; words already preprocessed and separated by whitespace.
    """

    def __init__(self, source, max_sentence_length=10000, limit=None):
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in itertools.islice(self.source, self.limit):
                line = to_unicode(line.lower()).split()
                i = 0
                while i < len(line):
                    yield line[i: i + self.max_sentence_length]
                    i += self.max_sentence_length
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            if os.path.isdir(self.source):
                fileList = os.listdir(self.source)
                for file in fileList:
                    with smart_open(os.path.join(self.source, file)) as fin:
                        for line in itertools.islice(fin, self.limit):
                            line = to_unicode(line.lower()).split()
                            i = 0
                            while i < len(line):
                                yield line[i: i + self.max_sentence_length]
                                i += self.max_sentence_length
            else:
                with smart_open(self.source) as fin:
                    for line in itertools.islice(fin, self.limit):
                        line = to_unicode(line.lower()).split()
                        i = 0
                        while i < len(line):
                            yield line[i: i + self.max_sentence_length]
                            i += self.max_sentence_length


WORD_VOCAB_NAME = "word_vocab.pkl"
IDX2WD_NAME = "idx2wd.pkl"
CHAR_VOCAB_NAME = "char_vocab.pkl"

class Text_vocab_builder():
    """
    build vocab and save vocab and index2word in the given path
    """
    def __init__(self, min_word_count, max_vocab_size, sub_sample, vocab_path, sentences=None):
        self.vocab = {}
        self.char_vocab={}
        self.index2word = []
        self._min_count = min_word_count
        self._max_vocab_size = max_vocab_size
        self._sub_sampling = sub_sample
        self.vocab_size = 0
        self.corpus_words = 0
        self.corpus_sentences = 0

        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")

        self.build_vocab(sentences)
        pkl_save(self.vocab, os.path.join(vocab_path, WORD_VOCAB_NAME))
        pkl_save(self.index2word, os.path.join(vocab_path, IDX2WD_NAME))
        pkl_save(self.char_vocab, os.path.join(vocab_path, CHAR_VOCAB_NAME))

    def build_vocab(self, sentences):
        self.scan_vocab(sentences)
        self.scale_vocab()
        self.finalize_vocab()

    def get_corpus_num(self):
        return self.corpus_sentences

    def get_words_count(self):
        count = 0
        if len(self.vocab) > 1:
            for key in list(self.vocab):
                count += self.vocab[key].count
        else:
            raise Exception("The vocabulary has not been initialized!!")
        return count

    def scan_vocab(self, sentences, progress_per=5000000):
        print("Scan Vocab : Collecting all words and their counts")
        idx, n_words = 0, 0
        min_reduce = 1
        vocab = collections.defaultdict(int)
        char_vocab = collections.defaultdict(int)
        for idx, sentence in enumerate(sentences):
            if idx % progress_per == 0:
                print("PROGRESS: at sentence #%i, there are %i words in vocab\r" % (idx, len(vocab)), end="")
                sys.stdout.flush()
            for word in sentence:
                vocab[word] += 1
                for c in word:
                    char_vocab[c]+=1
            if self._max_vocab_size and len(vocab) > self._max_vocab_size:
                n_words += self.prune_vocab(vocab, min_reduce)
                min_reduce += 1

        n_words += sum(itervalues(vocab))
        print("char_vocab's size: %d, raw_vocab's size: %i ,  raw_corpus: %i words,  %i sentences" % (len(char_vocab),len(vocab), n_words, idx + 1))
        self.vocab_size = len(vocab)
        self.corpus_words = n_words
        self.corpus_sentences = idx + 1
        self.raw_vocab = vocab
        self.raw_char_vocab = char_vocab

    def prune_vocab(self, vocab, min_reduce):
        # prune vocab to make sure that vocab_size <= max_vocab_size and return the number of discard words
        result = 0
        old_len = len(vocab)
        for w in list(vocab):
            if vocab[w] <= min_reduce:  # vocab[w] <= min_reduce:
                result += vocab[w]
                del vocab[w]
        print("pruned out %i tokens with count <=%i (before %i, after %i)\r" % (
            old_len - len(vocab), min_reduce, old_len, len(vocab)), end="")
        return result

    def scale_vocab(self):
        # Discard words less-frequent than min_count
        drop_unique, drop_total = 0, 0
        retain_total, retain_words = 0, []
        sample = self._sub_sampling
        char_list = []

        for c, _ in iteritems(self.raw_char_vocab):
            self.char_vocab[c] = len(char_list)
            char_list.append(c)

        for word, v in iteritems(self.raw_vocab):
            if v >= self._min_count:
                self.vocab[word] = Vocab(count=v, index=len(retain_words))
                retain_words.append(word)
                retain_total += v
            else:
                drop_unique += 1
                drop_total += v

        original_unique_total = len(retain_words) + drop_unique
        retain_unique_pct = len(retain_words) * 100 / max(original_unique_total, 1)
        print("With min_count=%d , vocab: %i words (%i%% of original %i, drops %i)" % (
            self._min_count, len(retain_words), retain_unique_pct, original_unique_total, drop_unique), end="")

        self.vocab_size = len(retain_words)
        self.index2word = retain_words

        original_total = retain_total + drop_total
        retain_pct = retain_total * 100 / max(original_total, 1)
        print("; corpus: %i word (%i%% of original %i, drops %i)"
              % (retain_total, retain_pct, original_total, drop_total))

        self.corpus_words = retain_total

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + np.sqrt(5)) / 2)

        downsample_total = 0
        for w in retain_words:
            v = self.raw_vocab[w]
            word_probability = (np.sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v

            self.vocab[w].sample_int = int(round(word_probability * 2 ** 32))

        self.raw_vocab = collections.defaultdict(int)

        print("Downsampling will leave estimated %i words in corpus (%.1f%% of prior %i)" % (
        downsample_total, downsample_total * 100.0 / max(retain_total, 1), retain_total))

        logger.info("Finally: (vocab's size: %i ),  (corpus: %i words,  %i sentences)", self.vocab_size,
                    self.corpus_words, self.corpus_sentences)

    def finalize_vocab(self):
        self.sort_vocab()

    def sort_vocab(self):
        # 词频数越大，在字典的位置越靠前
        self.index2word.sort(key=lambda word: self.vocab[word].count, reverse=True)
        for i, word in enumerate(self.index2word):
            self.vocab[word].index = i


SKIPGRAM = 1
CLUSTER_POINT = 2


class Batch_generator():
    def __init__(self, vocab_path, idx2word_path, sentence, window_size, negative_sample, epoch, min_sentence_length,
                 batch_save_path=None):
        self.vocab = pkl_load(vocab_path)
        self.idx2word = pkl_load(idx2word_path)
        self.sentences = sentence
        self.window_size = window_size
        self.save_path = batch_save_path
        self.negative_sample = negative_sample
        self.epoch = epoch
        self.min_sentence_length = min_sentence_length

    def skipGram_pair(self, word_idxs):
        """
        generate word pair for skip gram model_v
        :param word_idxs: sentence with words ids eg.[12,34,4,54,65732,2345,6]
        :return: [[1,2],[2,3]]
        """
        pair_list = []
        for i in range(0, len(word_idxs)):
            pos1, pos2 = max(i - self.window_size, 0), min(i + self.window_size, len(word_idxs) - 1)
            while pos1 < i:
                pair_list.append([word_idxs[i], word_idxs[pos1]])
                pos1 += 1
            while pos2 > i:
                pair_list.append([word_idxs[i], word_idxs[pos2]])
                pos2 -= 1

        return pair_list

    def cluster_point_pair(self, words_list):
        """
         pair : [target_word,rel_pair,unrel_pair]
         [[1,[2,3][4,5]],[2,[3,4][5,6]],...]
        """
        pair_list = []
        for i in range(0, len(words_list)):
            pos1, pos2 = max(i - self.window_size, 0), min(i + self.window_size, len(words_list) - 1)
            rel_pair = list(set(words_list[pos1:pos2 + 1]))
            rel_pair.remove(words_list[i])
            unrel_pair = []
            while (len(unrel_pair) < self.negative_sample * self.window_size):
                temp_word = random.sample(self.idx2word, 1)[0]
                temp_index = self.vocab[temp_word].index
                if temp_index not in unrel_pair and temp_index not in words_list:
                    unrel_pair.append(temp_index)

            if len(rel_pair) > 0 and len(unrel_pair) > 0:
                pair = [words_list[i], rel_pair, unrel_pair]
                pair_list.append(pair)
        return pair_list

    def generate_train_batch(self, generate_type):
        sentences = RepeatCorpusNTimes(self.sentences, self.epoch)
        for sentence in sentences:
            batch = []
            word_vocabs = [self.vocab[w].index for w in sentence if
                           w in self.vocab and self.vocab[w].sample_int > np.random.rand() * 2 ** 32]
            if self.min_sentence_length < len(word_vocabs):
                if generate_type == CLUSTER_POINT:
                    batch = self.cluster_point_pair(word_vocabs)
                if generate_type == SKIPGRAM:
                    batch = self.skipGram_pair(word_vocabs)
            if len(batch) > 0:
                yield batch


class Batch_estimator():
    def __init__(self, sentences, vocab_path, window_size, iter_num):
        self.sentences = sentences
        self.vocab = pkl_load(vocab_path)
        self.window_size = window_size
        self.epoch = iter_num

    def estimate_batch_num(self, model_type):
        """
        """
        total = 0
        for sentence in self.sentences:
            word_vocabs = [self.vocab[w].index for w in sentence if
                           w in self.vocab and self.vocab[w].sample_int > np.random.rand() * 2 ** 32]
            if model_type == SKIPGRAM:
                # total += self.batch_per_sentence_v1(len(word_vocabs), self.window_size)
                total += 1
            elif model_type == CLUSTER_POINT:
                total += len(word_vocabs)

        print('The total number of batches is around ', total * self.epoch)
        return total * self.epoch

    # def batch_per_sentence_v3(self, sentence_length, window_size):
    #     return (sentence_length - window_size) * window_size + (window_size * (window_size - 1) / 2)

    def batch_per_sentence_v1(self, sentence_length, window_size):
        num = 0
        begin, end = 0, sentence_length
        for pos in range(begin, end):
            num = num + pos + min(end - 1 - pos, window_size)
        return num

