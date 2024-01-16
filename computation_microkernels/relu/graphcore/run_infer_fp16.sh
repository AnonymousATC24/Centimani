#/bin/sh

#    for w, h, c,  n in config:
#       (7, 7, 32, 8)
time -p python relu.py --kernel 1 --input-width 7 --input-height 7 --channel-size 32 --batch-size 262_144 --dtype FLOAT16 --batches-per-step 1


#       (14,14,128,4)
time -p python relu.py --kernel 2 --input-width 14 --input-height 14 --channel-size 128 --batch-size 16_384 --dtype FLOAT16 --batches-per-step 1


#       (54, 54, 1024, 256, 1024)
time -p python relu.py --kernel 3 --input-width 54 --input-height 54 --channel-size 1024 --batch-size 128 --dtype FLOAT16 --batches-per-step 1


#       (128, 128, 128, 1024)
time -p python relu.py --kernel 4 --input-width 128 --input-height 128 --channel-size 128 --batch-size 128 --dtype FLOAT16 --batches-per-step 1
