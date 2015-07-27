text generation and summarization

Dependencies
 * [Keras](https://github.com/fchollet/keras/) 
 * [Sumy](https://github.com/miso-belica/sumy)


### Summarization (Sumy)

    python summarize.py --url https://en.wikipedia.org/wiki/George_W._Bush --num_sentences 10
	

### Text generation (Keras)

	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python generate.py --name patriotAct --url http://www.genekogan.com/txt/patriotAct.txt --nb_epochs 1 --max_epochs 50 --rnn_size 512 --num_layers 2 

### Notes

for Keras, large texts are good to offload to GPU.
	
	export PATH=/usr/local/cuda/bin:$PATH
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python lstm_text_generation.py
	
### Amazon EC2

[Amazon EC2 AMI](http://thecloudmarket.com/image/ami-b5b642f1--june12-keras-install-with-theano-and-gpu-support) for running Keras on AWS. Best to use an instance with high GPU performance (e.g. g2.8xlarge works well for me).