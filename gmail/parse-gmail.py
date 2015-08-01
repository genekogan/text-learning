import json
import ijson
import time
import re

def removeNonAscii(s): 
	return "".join(filter(lambda x: ord(x)<128, s))

def main_old():
	f2 = open('/Users/Gene/Data/email3.json','w')
	s = ''
	idx = 0
	with open('/Users/gene/Data/email.json') as f:
	    for line in f:
			if idx < 100000:
				s += line
				f2.write(line)
				idx+=1
			else:
				break
	f2.close()
	s=''
	with open('/Users/gene/Data/email2.json') as f:
	    for line in f:
			s += line
	JSON_Datalist = s
	the_dict = json.loads(JSON_Datalist)




def parse_email_content(content):
	# remove signature
	content = re.sub('--.+\nGene Kogan.+\nwebsite.+\|\|.+vimeo.+\n.+\|\|.+soundcloud.+\|\|.+flickr.+\n.+\|\|.+github.+\n.+\|\|.+twitter.+', '', content)
	#remove forwarded messages	
	if len(re.findall(r'---------- Forwarded message ----------.+', content)) > 0:
		for q in re.finditer(r'---------- Forwarded message ----------.+', content):
			content = content[0:q.start(0)]
	#remove quoted messages	
	if len(re.findall(r'\nOn |Mon|Tue|Wed|Thu|Fri|Sat|Sun| .*wrote:.+', content)) > 0:
		for q in re.finditer(r'\nOn |Mon|Tue|Wed|Thu|Fri|Sat|Sun| .*wrote:.+', content):
			content = content[0:q.start(0)]
	return content
	
def main(filepath, outputpath):
	include_from = 'Gene Kogan <kogan.gene@gmail.com>'
	exclude_to = 'GeneKogan<kogan.gene@gmail.com>'

	output_file = open(outputpath, 'w')	
	input_file = open(filepath)	
#	num_items = len(ijson.items(input_file, 'item'))

	num_found_items = 0
	errors = 0
	for (i,t) in enumerate(ijson.items(input_file, 'item')):
		if (i % 100 == 0):
			print "try email %d, found %d so far" % (i, num_found_items)
		try:
			if t['From'] == include_from and str(t['To'][0]) != exclude_to: 
				for p in t['parts']:
					if p['contentType']=='text/plain':
						content = parse_email_content(t["parts"][0]["content"])
						output_file.write(removeNonAscii(content))
						output_file.flush()
						num_found_items += 1
		except:
			print "Ooops, error "+str(errors)+"...."
			errors += 1
	output_file.close()
	input_file.close()
	
	
main('/Users/gene/Data/email.json', '/Users/gene/Code/Python/text-learning/chainer/data/myGmail/input.txt')