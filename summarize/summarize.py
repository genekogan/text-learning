from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import argparse

def main(url, num_sentences=10, language='english'):
	parser = HtmlParser.from_url(url, Tokenizer(language))
	stemmer = Stemmer(language)
	summarizer = Summarizer(stemmer)
	summarizer.stop_words = get_stop_words(language)
	for sentence in summarizer(parser.document, num_sentences):
		print(sentence)
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Summarize a URL')
	parser.add_argument('-u','--url', help='url', required=True)	
	parser.add_argument('-n','--num_sentences',help='Number of sentences. Default: 10', type=int, required=False)
	parser.add_argument('-l','--language',help='Language. Default: english', required=False)
	args = parser.parse_args()
	nargs = len(vars(args))
	if nargs == 4:
		main(args.url, args.num_sentences, args.language)
	elif nargs == 3:
		main(args.url, args.num_sentences)
	elif nargs == 2:
		main(args.url)
