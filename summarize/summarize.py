from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

presidents = ['George Washington','John Adams','Thomas Jefferson','James Madison','James Monroe','John Quincy Adams','Andrew Jackson','Martin Van Buren','William H. Harrison','John Tyler','James K. Polk','Zachary Taylor','Millard Fillmore','Franklin Pierce','James Buchanan','Abraham Lincoln','Andrew Johnson','Ulysses S. Grant','Rutherford B. Hayes','James A. Garfield','Chester A. Arthur','Grover Cleveland','Benjamin Harrison','Grover Cleveland','William McKinley','Theodore Roosevelt','William Howard Taft','Woodrow Wilson','Warren G. Harding','Calvin Coolidge','Herbert Hoover','Franklin D. Roosevelt','Harry S Truman','Dwight D. Eisenhower','John F. Kennedy','Lyndon B. Johnson','Richard M. Nixon','Gerald R. Ford','Jimmy Carter','Ronald Reagan','George Bush','William J. Clinton','George W. Bush','Barack Obama']

for p in reversed(presidents):
	url = "https://en.wikipedia.org/wiki/"+p.replace(' ','_')
	parser = HtmlParser.from_url(url, Tokenizer('english'))
	stemmer = Stemmer('english')
	summarizer = Summarizer(stemmer)
	summarizer.stop_words = get_stop_words('english')
	print(str(p)+'\n=====================')
	print(summarizer(parser.document, 1))
	print(str('\n'))

