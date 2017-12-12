import sys
import os
import time
import numpy as np
import collections
from six.moves import cPickle

UNKNOWN_CHAR = '*'
MAX_LENGTH = 275
MIN_LENGTH = 10
max_words = 300000

class Data:
    def __init__(self, data_dir, input_file, vocab_file, tensor_file):
        self.batch_size = 64
        self.unknow_char = UNKNOWN_CHAR
        input_file = os.path.join(data_dir, input_file)
        vocab_file = os.path.join(data_dir, vocab_file)
        tensor_file = os.path.join(data_dir, tensor_file)
        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.texts_vector = np.load(tensor_file)

    def preprocess(self, input_file, vocab_file, tensor_file):
        def handle(line):
            if len(line) > MAX_LENGTH:
                index_end = line.rfind('。', 0, MAX_LENGTH)
                index_end = index_end if index_end > 0 else MAX_LENGTH
                line = line[:index_end + 1]
            return line
        self.texts = [line.strip().replace('\n', '') for line in
                        open(input_file, encoding='utf-8')]
        self.texts = [handle(line) for line in self.texts if len(line) > MIN_LENGTH]
        # 所有字
        words = []
        for text in self.texts:
            words += [word for word in text]
        counter = collections.Counter(words)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        words, _ = zip(*count_pairs)

        # 取出现频率最高的词的数量组成字典，不在字典中的字用'*'代替
        words_size = min(max_words, len(words))
        self.words = words[:words_size] + (UNKNOWN_CHAR,)
        self.words_size = len(self.words)

        self.vocab = dict(zip(self.words, range(len(self.words))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)
        # self.tensor = np.array(list(map(self.vocab.get, data)))  

        # 字映射成id
        # self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        # self.id2char_dict = {i: w for i, w in enumerate(self.words)}
        # self.unknow_char = self.char2id_dict.get(UNKNOWN_CHAR)
        # self.char2id = lambda char: self.char2id_dict.get(char, self.unknow_char)
        # self.id2char = lambda num: self.id2char_dict.get(num)
        # self.texts = sorted(self.texts, key=lambda line: len(line))
        self.texts_vector = np.array([
            list(map(self.vocab.get, poetry)) for poetry in self.texts])
        np.save(tensor_file, self.texts_vector)

    def create_batches(self):
        self.n_size = len(self.texts_vector) // self.batch_size
        assert self.n_size > 0, 'data set is too small and need more data.'
        self.texts_vector = self.texts_vector[:self.n_size * self.batch_size]
        self.x_batches = []
        self.y_batches = []
        for i in range(self.n_size):
            batches = self.texts_vector[i * self.batch_size : (i + 1) * self.batch_size]
            length = max(map(len, batches))
            # 将长度不足的用 * 补充
            for row in range(self.batch_size):
                if len(batches[row]) < length:
                    r = length - len(batches[row])
                    batches[row][len(batches[row]) : length] = [self.vocab[self.unknow_char]] * r
            xdata = np.array(list(map(lambda x: np.array(x), batches)))
            ydata = np.copy(xdata)
            # 将标签整体往前移动一位， 代表当前对下一个的预测值            
            ydata[:, :-1] = xdata[:, 1:]
            ydata[:, -1] = xdata[:, 0]
            self.x_batches.append(xdata)
            self.y_batches.append(ydata)