import sys
import os
import time
import numpy as np
import collections
from six.moves import cPickle
import data_loader
import model

BEGIN_CHAR = '<'
END_CHAR = '>'
UNKNOWN_CHAR = '*'
MAX_LENGTH = 280
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
        print(self.words_size)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.texts_vector = np.load(tensor_file)
        self.words_size = len(self.chars)

    def preprocess(self, input_file, vocab_file, tensor_file):
        def handle(line):
            if len(line) > MAX_LENGTH:
                index_end = line.rfind('。', 0, MAX_LENGTH)
                index_end = index_end if index_end > 0 else MAX_LENGTH
                line = line[:index_end + 1]
            return BEGIN_CHAR + line + END_CHAR
        self.texts = [line.strip().replace('\n', '') for line in
                        open(input_file, encoding='utf-8')]
        self.texts = [handle(line) for line in self.texts if len(line) > MIN_LENGTH]
        # 所有字
        words = ['*', ' ']
        for text in self.texts:
            words += [word for word in text]
        self.words = list(set(words))
        self.words_size = len(self.words)

        self.vocab = dict(zip(self.words, range(len(self.words))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)
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