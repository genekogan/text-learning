import nltk
from nltk.model import NgramModel
from nltk.probability import LidstoneProbDist
from nltk.tokenize import WhitespaceTokenizer
import argparse

parser = argparse.ArgumentParser(description='Perplexity-based pruning of generated text')
parser.add_argument('--trainfile', required=True, type=str, help='training file')
parser.add_argument('--testfile', required=True, type=str, help='testing file')
parser.add_argument('--num_grams', default=4, type=int, help='num grams for estimator')
parser.add_argument('--estimator_probability', default=0.2, type=float, help='Lidstone estimator probability')
parser.add_argument('--cutoff_max_perplexity', default=20, type=float, help='running cutoff max perplexity')
parser.add_argument('--output_max_perplexity', default=10, type=float, help='output max perplexity')
parser.add_argument('--min_sentence_length', default=8, type=int, help='min sentence length')

args = parser.parse_args()

f_train = open(args.trainfile)
f_test = open(args.testfile)
training_raw = f_train.read()
testing_raw = f_test.read()

#train = [word.lower() for word in nltk.word_tokenize(training_raw)]
#test = [word.lower() for word in nltk.word_tokenize(testing_raw)]

training_spans = WhitespaceTokenizer().span_tokenize(training_raw)
spans = [span for span in training_spans]
training_offsets = [span[0] for span in spans]
train = []
for s in spans:
	train.append(training_raw[s[0]:s[1]])

testing_spans = WhitespaceTokenizer().span_tokenize(testing_raw)
spans = [span for span in testing_spans]
testing_offsets = [span[0] for span in spans]
test = []
for s in spans:
	test.append(testing_raw[s[0]:s[1]])

estimator = lambda fdist, bins: LidstoneProbDist(fdist, args.estimator_probability) 
lm = NgramModel(args.num_grams, train, estimator=estimator)

t0 = 0
t1 = 1
current_best=''
while t1 < len(test):
	perplexity = lm.perplexity(test[t0:t1])
	if perplexity > args.cutoff_max_perplexity:
		if (len(current_best)>1):
			print current_best+'.'
			current_best=''
		t0 = t1 + 1
		t1 = t0 + 1
	else:
		t1 += 1
		if t1-t0 > args.min_sentence_length and perplexity < args.output_max_perplexity:
			current_best = testing_raw[testing_offsets[t0]:testing_offsets[t1+1]]
			#current_best = ' '.join(test[t0:t1])
			
	
#print "perplexity "+str(t0)+":"+str(t1)+" = "+str(perplexity)