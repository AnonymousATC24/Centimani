#/bin/sh

#    for w, h, c,  n, k,  s, r, pad_w, pad_h, wstride, hstride in config:
#       (7, 7, 32, 8, 32, 3, 3, 0,     0,     1,       1),
time -p python conv2d.py --kernel 1 --mode infer --input-width 7 --input-height 7 --channel-size 32 --batch-size 8192 --filter-number 32 --kernel-size 3 --padding 0 --stride 1 --dtype FLOAT
time -p python conv2d.py --kernel 1 --mode train --input-width 7 --input-height 7 --channel-size 32 --batch-size 8192 --filter-number 32 --kernel-size 3 --padding 0 --stride 1 --dtype FLOAT


#       (14,14,128,4, 256,3, 3, 1,     1,     1,       1),
time -p python conv2d.py --kernel 2 --mode infer --input-width 14 --input-height 14 --channel-size 128 --batch-size 128 --filter-number 256 --kernel-size 3 --padding 1 --stride 1 --dtype FLOAT
time -p python conv2d.py --kernel 2 --mode train --input-width 14 --input-height 14 --channel-size 128 --batch-size 128 --filter-number 256 --kernel-size 3 --padding 1 --stride 1 --dtype FLOAT


#       (54, 54, 1024, 256, 1024, 3, 3, 1, 1, 1, 1), # DeepSpeech
time -p python conv2d.py --kernel 3 --mode infer --input-width 54 --input-height 54 --channel-size 1024 --batch-size 2 --filter-number 1024 --kernel-size 3 --padding 1 --stride 1 --dtype FLOAT
time -p python conv2d.py --kernel 3 --mode train --input-width 54 --input-height 54 --channel-size 1024 --batch-size 2 --filter-number 1024 --kernel-size 3 --padding 1 --stride 1 --dtype FLOAT


#       (128, 128, 128, 1024, 128, 5, 5, 0, 0, 1, 1), # resnet50  
time -p python conv2d.py --kernel 4 --mode infer --input-width 128 --input-height 128 --channel-size 128 --batch-size 4 --filter-number 128 --kernel-size 5 --padding 0 --stride 1 --dtype FLOAT
time -p python conv2d.py --kernel 4 --mode train --input-width 128 --input-height 128 --channel-size 128 --batch-size 4 --filter-number 128 --kernel-size 5 --padding 0 --stride 1 --dtype FLOAT
