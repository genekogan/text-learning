text generation and summarization

Dependencies
 * [Keras](https://github.com/fchollet/keras/) 
 * [Sumy](https://github.com/miso-belica/sumy)


### Summarization

    python summarize.py --url https://en.wikipedia.org/wiki/George_W._Bush --num_sentences 10
	

### notes

for Keras, large texts are good to offload to GPU, e.g.
	
	export PATH=/usr/local/cuda/bin:$PATH
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python lstm_text_generation.py