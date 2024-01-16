#!/bin/sh

time -p python ./lstm_fp16.py --timesteps 50 --batch-size 64 --hidden-size 256
time -p python ./lstm_fp16.py --timesteps 25 --batch-size 32 --hidden-size 512
time -p python ./lstm_fp16.py --timesteps 25 --batch-size 16 --hidden-size 512
time -p python ./lstm_fp16.py --timesteps 50 --batch-size 32 --hidden-size 512

