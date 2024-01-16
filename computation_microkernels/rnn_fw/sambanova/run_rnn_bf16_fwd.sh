#!/bin/sh

time -p python rnn_fwd.py --dtype bfloat16 --time-step=50 --batch-size=64 --input-size=256 --hidden-size=256
time -p python rnn_fwd.py --dtype bfloat16 --time-step=25 --batch-size=32 --input-size=512 --hidden-size=512
time -p python rnn_fwd.py --dtype bfloat16 --time-step=25 --batch-size=16 --input-size=512 --hidden-size=512
time -p python rnn_fwd.py --dtype bfloat16 --time-step=50 --batch-size=32 --input-size=512 --hidden-size=512
