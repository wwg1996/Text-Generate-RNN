# coding:utf-8

import argparse
import sys
import os
import time
import numpy as np
import tensorflow as tf
from model import Model
from data_loader import Data

BEGIN_CHAR = '<'
END_CHAR = '>'
UNKNOWN_CHAR = '*'
epochs = 50

data_dir = 'data/poetry/'
input_file = 'tang(simplified).txt'
vocab_file = 'vocab_tang(simplified).pkl'
tensor_file = 'tensor_tang(simplified).npy'

model_dir = 'model/poetry/'


def train(data, model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        n = 0
        for epoch in range(epochs):
            sess.run(tf.assign(model.learning_rate, 0.002 * (0.97 ** epoch)))
            pointer = 0
            for batche in range(data.n_size):
                n += 1
                feed_dict = {model.x_tf: data.x_batches[pointer], model.y_tf: data.y_batches[pointer]}
                pointer += 1
                train_loss, _, _ = sess.run([model.cost, model.final_state, model.train_op], feed_dict=feed_dict)
                sys.stdout.write('\r')
                info = "{}/{} (epoch {}) | train_loss {:.5f}" \
                    .format(epoch * data.n_size + batche,
                            epochs * data.n_size, epoch, train_loss)
                sys.stdout.write(info)
                # sys.stdout.flush()
                if (epoch * data.n_size + batche) % 1000 == 0 \
                        or (epoch == epochs-1 and batche == data.n_size-1):
                    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=n)
                    sys.stdout.write('\n')
                    print("model saved to {}".format(checkpoint_path))
            sys.stdout.write('\n')


def sample(data, model, head=u''):
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sa = int(np.searchsorted(t, np.random.rand(1) * s))
        return data.id2char(sa)

    for word in head:
        if word not in data.words:
            return u'{} 不在字典中'.format(word)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, model_file)

        if head:
            print('生成藏头诗 ---> ', head)
            poem = BEGIN_CHAR
            for head_word in head:
                poem += head_word
                x = np.array([list(map(data.char2id, poem))])
                state = sess.run(model.cell.zero_state(1, tf.float32))
                feed_dict = {model.x_tf: x, model.initial_state: state}
                [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
                word = to_word(probs[-1])
                while word != u'，' and word != u'。':
                    poem += word
                    x = np.zeros((1, 1))
                    x[0, 0] = data.char2id(word)
                    [probs, state] = sess.run([model.probs, model.final_state],
                                              {model.x_tf: x, model.initial_state: state})
                    word = to_word(probs[-1])
                poem += word
            return poem[1:]
        else:
            poem = ''
            head = BEGIN_CHAR
            x = np.array([list(map(data.char2id, head))])
            state = sess.run(model.cell.zero_state(1, tf.float32))
            feed_dict = {model.x_tf: x, model.initial_state: state}
            [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
            word = to_word(probs[-1])
            while word != END_CHAR:
                poem += word
                x = np.zeros((1, 1))
                x[0, 0] = data.char2id(word)
                [probs, state] = sess.run([model.probs, model.final_state],
                                          {model.x_tf: x, model.initial_state: state})
                word = to_word(probs[-1])
            return poem


def main():
    msg = """
            Usage:
            Training: 
                python poetry_gen.py --mode train
            Sampling:
                python poetry_gen.py --mode sample --head 明月别枝惊鹊
            """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='sample',
                        help=u'usage: train or sample, sample is default')
    parser.add_argument('--head', type=str, default='',
                        help='生成藏头诗')

    args = parser.parse_args()

    if args.mode == 'sample':
        infer = True  # True
        data = Data(data_dir, input_file, vocab_file, tensor_file)
        model = Model(data=data, infer=infer)
        print(sample(data, model, head=args.head))
    elif args.mode == 'train':
        infer = False
        data = Data(data_dir, input_file, vocab_file, tensor_file)
        model = Model(data=data, infer=infer)
        print(train(data, model))
    else:
        print(msg)


if __name__ == '__main__':
    main()