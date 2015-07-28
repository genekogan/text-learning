from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
from keras.optimizers import RMSprop
import numpy as np
import random, sys
import time
import argparse
import cPickle

'''
	LSTM text generation
'''

parser = argparse.ArgumentParser(description='Train a character-level language model')
parser.add_argument('--name', required=True, type=str, help='name')
parser.add_argument('--url', required=True, type=str, help='url for text')
parser.add_argument('--rnn_size', default=512, type=int, help='size of LSTM internal state')
parser.add_argument('--num_layers', default=2, type=int, help='number of layers in the LSTM')
parser.add_argument('--learning_rate',default=2e-3, type=float, help='learning rate')
parser.add_argument('--decay_rate',default=0.95, type=float, help='decay rate for rmsprop')
parser.add_argument('--dropout',default=0, type=float, help='dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
parser.add_argument('--seq_length',default=20, type=int, help='number of timesteps to unroll for')
parser.add_argument('--seq_step',default=3, type=int, help='number of steps between sequences, default to seq_length')
parser.add_argument('--batch_size',default=128, type=int, help='number of sequences to train on in parallel')
parser.add_argument('--max_epochs',default=50, type=int, help='number of full passes through the training data')
parser.add_argument('--nb_epochs',default=1, type=int, help='number of epochs per iteration')
parser.add_argument('--grad_clip',default=5, type=float, help='clip gradients at this value')
parser.add_argument('--seed',default=0, type=int, help='numpy manual random number generator seed')

args = parser.parse_args()

if args.seq_step is None:
    args.seq_step = args.seq_length

# writing outputs
filename = args.name+'_rnn'+str(args.rnn_size)+'_layers'+str(args.num_layers)+'_seqlen'+str(args.seq_length)+'_batch'+str(args.batch_size)+'_epochs'+str(args.max_epochs)+'_'+str(args.nb_epochs)
generated_text_file = open('output/'+filename+'.txt','w')


path = get_file(args.name, origin=args.url)
text = open(path).read().lower()
print('corpus length:', len(text))

chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = args.seq_length
step = args.seq_step
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

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
for layer in range(0, args.num_layers):
    input_size = len(chars) if layer == 0 else args.rnn_size
    return_sequences = (layer + 1) < args.num_layers
    model.add(LSTM(input_size, args.rnn_size, return_sequences=return_sequences))
    if args.dropout: model.add(Dropout(args.dropout))
model.add(Dense(args.rnn_size, len(chars)))
model.add(Activation('softmax'))

rmsprop = RMSprop(lr=args.learning_rate, rho=args.decay_rate, clipnorm=args.grad_clip)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

# helper function to sample an index from a probability array
def sample(a, diversity=0.75):
    if random.random() > diversity:
        return np.argmax(a)
    while 1:
        i = random.randint(0, len(a)-1)
        if a[i] > random.random():
            return i

# train the model, output generated text after each iteration
for iteration in range(1, 1+args.max_epochs):
	print()
	print('-' * 50)
	print('Iteration', iteration)

	start_time = time.time()
	model.fit(X, y, batch_size=args.batch_size, nb_epoch=args.nb_epochs)
	elapsed_time = time.time() - start_time
	
	start_index = random.randint(0, len(text) - maxlen - 1)

	for diversity in [0.2, 0.5, 1.0, 1.25]:
		print()
		print('----- diversity:', diversity)

		generated = ''
		sentence = text[start_index : start_index + maxlen]
		generated += sentence
		print('----- Generating with seed: "' + sentence + '"')
		sys.stdout.write(generated)

		for iteration2 in range(400 * iteration ):
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
		generated_text_file.write('\n\n\niteration '+str(iteration)+', diversity '+str(diversity)+', elapsed '+str(elapsed_time)+'\n=================================================\n\n')
		generated_text_file.write(generated+'\n')
		generated_text_file.flush()
		model_file = file('obj.save', 'wb')
		cPickle.dump(model, model_file, protocol=cPickle.HIGHEST_PROTOCOL)
		model_file.close()

generated_text_file.close()
