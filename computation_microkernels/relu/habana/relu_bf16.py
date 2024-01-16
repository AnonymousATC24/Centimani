import numpy as np
import torch
import time

device = torch.cuda.current_device()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

config = (
    #    w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride, hstride
#    (700, 161, 1, 4, 32, 20, 5, 0, 0, 2, 2), 
#    (700, 161, 1, 32, 32, 20, 5, 0, 0, 2, 2), 
#    (341, 79, 32, 4, 32, 10, 5, 0, 0, 2, 2), 
#    (341, 79, 32, 32, 32, 10, 5, 0, 0, 2, 2), 
#    (480, 48, 1, 16, 16, 3, 3, 1, 1, 1, 1),
#    (240, 24, 16, 16, 32, 3, 3, 1, 1, 1, 1),
#    (120, 12, 32, 16, 64, 3, 3, 1, 1, 1, 1),
#    (60, 6, 64, 16, 128, 3, 3, 1, 1, 1, 1),
#    (108, 108, 3, 8, 64, 3, 3, 1, 1, 2, 2),
#    (54, 54, 64, 8, 64, 3, 3, 1, 1, 1, 1),
#(27, 27, 128, 8, 128, 3, 3, 1, 1, 1, 1),
#112	112	64	16	64	3	3	0	0	1	1 # resnet50
#7	7	512	8	2048	1	1	0	0	1	1 # resnet50
#108	108	3	8	64	3	3	1	1	2	2 # Face Recognition
#700	161	1	16	64	5	5	1	1	2	2 # DeepSpeech
#480	48	1	16	16	3	3	1	1	1	1 # OCR
#(128, 128, 1024, 128, 1024, 5, 5, 0, 0, 1, 1), # resnet50  200Tflops
#(7, 7, 2048, 1024, 2048, 3, 3, 0, 0, 1, 1), # resnet50
#(54, 54, 1024, 256, 1024, 3, 3, 1, 1, 1, 1), # DeepSpeech
#(700, 161, 32, 512, 32, 20, 20, 0, 0, 2, 2), # Face Recognition
#(480, 48, 16, 1024, 16, 3, 3, 1, 1, 1, 1), # OCR
#(14, 14, 256, 256, 256, 1, 1, 0, 0, 1, 1),
#(14,14,128,4,256,3,3,1,1,1,1),
#(7, 7, 32, 8, 32, 3, 3, 0, 0, 1, 1),
(7, 7, 32, 262144, 32, 3, 3, 0, 0, 1, 1),
(14,14,128,16384,256,3,3,1,1,1,1),
(54, 54, 1024, 128, 1024, 3, 3, 1, 1, 1, 1), # DeepSpeech
(128, 128, 128, 128, 128, 5, 5, 0, 0, 1, 1), # resnet50  200Tflops
)

print("| w, h, c, n, Dtype, TFLOPS_FORW |")
print("|--------------------------------|")

#for dataType in (torch.float32, torch.float16, torch.bfloat16): #np.float32, np.float16,
for dataType in (torch.bfloat16,):

    for w, h, c, n, k, s, r, pad_w, pad_h, wstride, hstride in config:

        dim_input = (n, c, w, h)

        input_tensor = torch.rand(dim_input, dtype = dataType).to(device)
        layer = torch.nn.ReLU().to(device)

        n_iters = 100
        n_warmup = 5
        #warm-up
        for loop in range(n_warmup):
            result = layer(input_tensor)

        forward_pass_time = []
        
        for loop in range(n_iters):
            start_time_forward = 0.0
            end_time_forward = 0.0
            start_time_forward = time.time()
            result = layer(input_tensor)
            torch.cuda.synchronize()
            end_time_forward = time.time()

            forward_pass_time.append(end_time_forward - start_time_forward)

        tflops = 2 * w * h * c * n
        tflops_forw = tflops/(sum(forward_pass_time)/n_iters)/10**12 #tflops

        print("A100,Training,%s,%d,100,1,%d,%d,%d,0.0,%f,%f,%f,%f" % (input_tensor.dtype, n, c, w, h, (sum(forward_pass_time)/n_iters)/n, n/(sum(forward_pass_time)/n_iters), tflops_forw, (sum(forward_pass_time)/n_iters)/n))
