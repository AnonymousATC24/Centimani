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
(7, 7, 32, 131072, 32, 3, 3, 0, 0, 1, 1),
(14,14,128,4096,256,3,3,1,1,1,1),
(54, 54, 1024, 16, 1024, 3, 3, 1, 1, 1, 1), # DeepSpeech
(128, 128, 128, 32, 128, 5, 5, 0, 0, 1, 1), # resnet50  200Tflops
#(7, 7, 32, 65536, 32, 3, 3, 0, 0, 1, 1),
#(14,14,128,1024,256,3,3,1,1,1,1),
#(54, 54, 1024, 8, 1024, 3, 3, 1, 1, 1, 1), # DeepSpeech
#(128, 128, 128, 16, 128, 5, 5, 0, 0, 1, 1), # resnet50  200Tflops
)

print("| w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride, hstride, Dtype, TFLOPS_FORW, TFLOPS_BACK|")
print("|---------------------------------------------------------------------------------------------------------|")

#for dataType in (torch.float32, torch.float16, torch.bfloat16): #np.float32, np.float16,
for dataType in (torch.bfloat16,):
#for dataType in (torch.float32,):

    for w, h, c, n, k, s, r, pad_w, pad_h, wstride, hstride in config:

        dim_input = (n, c, w, h)

        input_tensor = torch.rand(dim_input, dtype = dataType).to(device)
        grad_tensor = torch.ones(n, k, int((w-s+2*pad_w)/wstride) + 1, int((h-r+2*pad_h)/hstride) + 1).to(device)
        layer = torch.nn.Conv2d(c, k, (s, r), stride = (wstride, hstride), padding = (pad_w, pad_h), dtype = dataType).to(device)

        n_iters = 100
        n_warmup = 5
        #warm-up
        for loop in range(n_warmup):
            result = layer(input_tensor)
            result.backward(grad_tensor)

        forward_pass_time = []
        backward_pass_time = []
        
        for loop in range(n_iters):
            start_time_forward = 0.0
            end_time_forward = 0.0
            start_time_forward = time.time()
            result = layer(input_tensor)
            torch.cuda.synchronize()
            end_time_forward = time.time()

            start_time_backward = time.time()
            result.backward(grad_tensor)
            torch.cuda.synchronize()
            end_time_backward = time.time()
            
            forward_pass_time.append(end_time_forward - start_time_forward)
            backward_pass_time.append(end_time_backward - start_time_backward)

        w_0 = (w + 2*pad_w - s)/wstride + 1
        h_0 = (h + 2*pad_h - r)/hstride + 1
        tflops = 2 * (w_0*h_0) * s * r * c * k * n
        tflops_forw = tflops/(sum(forward_pass_time)/n_iters)/10**12 #tflops
        #tflops_forw = ((w / wstride * h / hstride) * s * r * c * k * n)/(sum(forward_pass_time)/n_iters)/10**12 #tflops
        tflops_back = tflops/(sum(backward_pass_time)/n_iters)/10**12 #tflops
        #tflops_back = 2*((w / wstride * h / hstride) * s * r * c * k * n)/(sum(backward_pass_time)/n_iters)/10**12 #tflops

        #print("|%5d|%5d|%3d|%3d|%3d|%3d|%3d|%3d|%3d|%3d|%3d|%14s|%10f" % (w, h, c, n, k, s, r, pad_w, pad_h, wstride, hstride, input_tensor.dtype, tflops_forw))
        print("A100,Training,%s,Conv2d_bckwd,%d,100,1,%d,%d,%d,%d,%d,%d,%d,0.0,%f,None,%f,%f,%f" % (input_tensor.dtype, n, w, h, c, k, s, pad_w, wstride, (sum(backward_pass_time)/n_iters)/n, n/(sum(backward_pass_time)/n_iters), tflops_back, (sum(backward_pass_time)/n_iters)/n))
