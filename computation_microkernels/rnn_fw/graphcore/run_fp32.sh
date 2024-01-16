#!/bin/sh

time -p python ./lstm_fp32.py --timesteps 50 --batch-size 32 --hidden-size 256
time -p python ./lstm_fp32.py --timesteps 25 --batch-size 16 --hidden-size 512
time -p python ./lstm_fp32.py --timesteps 25 --batch-size 8 --hidden-size 512
time -p python ./lstm_fp32.py --timesteps 50 --batch-size 16 --hidden-size 512

