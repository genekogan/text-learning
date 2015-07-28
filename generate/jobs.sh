#!/bin/bash 

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python generate.py --name patriotAct2 --url http://www.genekogan.com/txt/patriotAct.txt --nb_epochs 6 --max_epochs 20 --rnn_size 128 --batch_size 32 --num_layers 2 --seq_length 20 --seq_step 3 ; 
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python generate.py --name patriotAct2 --url http://www.genekogan.com/txt/patriotAct.txt --nb_epochs 6 --max_epochs 20 --rnn_size 512 --batch_size 32 --num_layers 2 --seq_length 20 --seq_step 3 



THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python generate.py --name patriotAct2 --url http://www.genekogan.com/txt/patriotAct.txt --nb_epochs 6 --max_epochs 20 --rnn_size 512 --batch_size 128 --num_layers 2 --seq_length 20 --seq_step 3
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python generate.py --name patriotAct2 --url http://www.genekogan.com/txt/patriotAct.txt --nb_epochs 6 --max_epochs 20 --rnn_size 512 --batch_size 128 --num_layers 2 --seq_length 50 --seq_step 5



