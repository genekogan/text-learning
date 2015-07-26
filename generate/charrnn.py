from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
from keras.optimizers import RMSprop
import numpy as np
import random, sys

'''
    Adapted from lstm_text_generation.py to follow char-rnn more closely.
'''

import argparse
parser = argparse.ArgumentParser(description='Train a character-level language model')
# parser.add_argument('--data_dir', default='data/tinyshakespeare', help='data directory. Should contain the file input.txt with input data')
parser.add_argument('--rnn_size', default=128, type=int, help='size of LSTM internal state')
parser.add_argument('--num_layers', default=2, type=int, help='number of layers in the LSTM')
# parser.add_argument('--model', default='lstm', help='for now only lstm is supported. keep fixed')
# optimization
parser.add_argument('--learning_rate',default=2e-3, type=float, help='learning rate')
# parser.add_argument('--learning_rate_decay',default=0.97, type=float, help='learning rate decay')
# parser.add_argument('--learning_rate_decay_after',default=10, type=int, help='in number of epochs, when to start decaying the learning rate')
parser.add_argument('--decay_rate',default=0.95, type=float, help='decay rate for rmsprop')
parser.add_argument('--dropout',default=0, type=float, help='dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
parser.add_argument('--seq_length',default=50, type=int, help='number of timesteps to unroll for')
parser.add_argument('--step',default=None, type=int, help='number of steps between sequences, default to seq_length')
parser.add_argument('--batch_size',default=128, type=int, help='number of sequences to train on in parallel')
parser.add_argument('--max_epochs',default=50, type=int, help='number of full passes through the training data')
parser.add_argument('--nb_epochs',default=1, type=int, help='number of epochs per iteration')
parser.add_argument('--grad_clip',default=5, type=float, help='clip gradients at this value')
parser.add_argument('--train_frac',default=0.95, type=float, help='fraction of data that goes into train set')
# test_frac will be computed as 1 - train_frac, there is no val_frac
# parser.add_argument('--init_from', default='', help='initialize network parameters from checkpoint at this path')
# bookkeeping
parser.add_argument('--seed',default=0, type=int, help='numpy manual random number generator seed')
# parser.add_argument('--print_every',default=1, type=int, help='how many steps/minibatches between printing out the loss')
# parser.add_argument('--eval_val_every',default=1000, type=int, help='every how many iterations should we evaluate on validation data?')
# parser.add_argument('--checkpoint_dir', default='cv', help='output directory where checkpoints get written')
# parser.add_argument('--savefile',default='lstm', help='filename to autosave the checkpont to. Will be inside checkpoint_dir/')
# GPU/CPU
# parser.add_argument('--gpuid',default=0, help='which gpu to use. -1 = use CPU')
args = parser.parse_args()

if args.step is None:
    args.step = args.seq_length

print('step size is', args.step)

np.random.seed(args.seed) # for reproducibility

path = get_file('patriotAct.txt', origin="http://genekogan.com/txt/patriotAct.txt")
#path = get_file('marquez.txt', origin="http://pauladaunt.com/books/MARQUES,%20Gabriel%20Garcia%20-%20One%20Hundred%20Years%20of%20Solitude.txt")
text = open(path).read()#.lower()
print('corpus length:', len(text))

chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = args.seq_length
step = args.step
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

split_point = int(len(X)*args.train_frac)
X_train = X[:split_point]
y_train = y[:split_point]
X_test = X[split_point:]
y_test = y[split_point:]

print('train:', X_train.shape[0], 'samples,', X_train.shape[0] / args.batch_size, 'batches')
print('test:', X_test.shape[0], 'samples,', X_test.shape[0] / args.batch_size, 'batches')

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
for layer in range(0, args.num_layers):
    # the first layer has input_size = len(chars)
    input_size = len(chars) if layer == 0 else args.rnn_size
    # the last layer has return_sequences=False
    return_sequences = (layer + 1) < args.num_layers
    model.add(LSTM(input_size, args.rnn_size, return_sequences=return_sequences))
    if args.dropout: model.add(Dropout(args.dropout))
model.add(Dense(args.rnn_size, len(chars)))
model.add(Activation('softmax'))

rmsprop = RMSprop(lr=args.learning_rate, rho=args.decay_rate, clipnorm=args.grad_clip)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

# helper function to sample an index from a probability array
def sample(a, temperature=1.0):
    a = np.log(a)/temperature
    a = np.exp(a)/np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1,a,1))

# train the model, output generated text after each iteration
for iteration in range(1, args.max_epochs):
    bb = iteration
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=args.batch_size, nb_epoch=args.nb_epochs,
        show_accuracy=True, verbose=1, validation_data=(X_test, y_test))

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index : start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for iteration2 in range(600*bb):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()