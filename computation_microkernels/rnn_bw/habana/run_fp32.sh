#!/bin/sh

time -p python rnn_fp32.py --time_step=50 --batch_size=32 --input_size=256 --hidden_size=256
time -p python rnn_fp32.py --time_step=25 --batch_size=16 --input_size=512 --hidden_size=512
time -p python rnn_fp32.py --time_step=25 --batch_size=8 --input_size=512 --hidden_size=512
time -p python rnn_fp32.py --time_step=50 --batch_size=16 --input_size=512 --hidden_size=512
