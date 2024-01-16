#

def processRawData():
    system = 'Graphcore'

    # Both headers are identical.
    headerLine = '<empty>'

    for dtype in ['fp16', 'fp32']:

        filepathIn = f'results_raw_{dtype}.txt'
        filepathOut = f'results_intermediate_{dtype}.csv'

        headerCount = 0

        with open(filepathIn) as fpIn:
            with open(filepathOut, 'w') as fpOut:
                line = fpIn.readline()

                while line:
                    if len(line) == 1:
                        line = fpIn.readline()
                        continue
                    if line.startswith('PopART'):
                        line = fpIn.readline()
                        continue
                    if line.startswith('Compiling'):
                        line = fpIn.readline()
                        continue
                    if line.startswith('Executing'):
                        line = fpIn.readline()
                        continue
                    if line.startswith('user'):
                        line = fpIn.readline()
                        continue
                    if line.startswith('sys'):
                        line = fpIn.readline()
                        continue
                    #(poptorch_env)
                    if line.startswith(' '):
                        line = fpIn.readline()
                        continue
                    #
                    if line.startswith('(poptorch_env)'):
                        line = fpIn.readline()
                        continue

                    #
                    if line.startswith(system):
                        line = line.rstrip() + ','

                    #real 57.08
                    if line.startswith('real'):
                        line = line.replace('real ','')
                        fpOut.write(line)
                        line = fpIn.readline()
                        continue

                    #Duration: 40.107 seconds
                    if line.startswith('Duration'):
                        line = line.replace('Duration: ','')
                        line = line.replace(' seconds',',')
                        line = line.rstrip()

                    if line.startswith('mfg'):
                        headerCount += 1
                        if headerCount > 1:
                            line = fpIn.readline()
                            continue

                        headerLine = line
                        fpOut.write(line)
                        line = fpIn.readline()
                        continue

                    #0.633    sec/itr.   647284.987433 items/sec,  74.829707 TFLOPS
                    if line.find('sec/itr.'):
                        line = line.replace('    sec/itr.   ', ',,')
                        line = line.replace('   sec/itr.   ', ',,')
                        line = line.replace(' items/sec,  ', ',')
                        line = line.replace(' TFLOPS', ',')
                        line = line.rstrip()


                    fpOut.write(line)
                    print(f"Line: {line}")


                    line = fpIn.readline()

    return headerLine.rstrip()


def calculateTFLOPS(headerLine):
    import pandas as pd

    for dtype in ['fp16', 'fp32']:

        filepathIn = f'results_intermediate_{dtype}.csv'
        chunksize = 2

        headerArray = headerLine.split(',')
        headerArray.append('h0')
        headerArray.append('w0')
        resultsFinal = [headerArray]

        # header=0 indicates the file contains a header.
        textFileReader = pd.read_csv(filepathIn, header=0, iterator=True, chunksize=chunksize)

        chunkNum = 0
        for chunk in textFileReader:
            print()
            print()
            print(chunk)

            inferIndex = chunkNum * chunksize
            trainIndex = inferIndex + 1

            chunk.loc[inferIndex, 'spare'] = 0

            # Calculate seconds for backwards step only.
            sec_per_iter = chunk.loc[trainIndex, 'sec_per_iter'] -  chunk.loc[inferIndex, 'sec_per_iter']
            chunk.loc[trainIndex, 'spare'] = sec_per_iter

            input_height = chunk.loc[trainIndex, 'Input_height']
            padding = chunk.loc[trainIndex, 'Padding']
            kernel_size = chunk.loc[trainIndex, 'Kernel_size']
            stride = chunk.loc[trainIndex, 'Stride']


            # h0 = (h + 2 * PADh - Fh) / STRIDEh + 1
            # h0 = (input-height + 2 * padding - kernel-size) / stride + 1
            h0 = (input_height + 2 * padding - kernel_size) / stride + 1
            chunk.loc[trainIndex, 'h0'] = h0

            input_width = chunk.loc[trainIndex, 'Input_width']
            w0 = (input_width + 2 * padding - kernel_size) / stride + 1
            chunk.loc[trainIndex, 'w0'] = w0

            kernel_num = chunk.loc[trainIndex, 'Kernel_num']
            channel = chunk.loc[trainIndex, 'Channel']
            kernel_size = chunk.loc[trainIndex, 'Kernel_size']
            batch_size = chunk.loc[trainIndex, 'Batch_size']
            batches_per_step = chunk.loc[trainIndex, 'Batches_per_Step']

            # w         = width             = (input-width)
            # h         = height            = (input-height)
            # c         = channels in       = (channel-size)
            # n         = batch size        = (batch-size)
            # k         = kernel output channels    = (filter-number) or (Kernel_num)
            # s         = filter width      = (kernel-size)
            # r         = filter height     = (kernel-size)
            # pad_w     = padding width     = (padding)
            # pad_h     = padding height    = (padding)
            # wstride   = stride width      = (stride)
            # hstride   = stride height     = (stride)

            # tflops = 2 * (w_0*h_0) * s * r * c * k * n
            # tflops = 2 * (w_0*h_0) * kernel-size * kernel-size * channel-size * filter-number * batch-size
            chunk.loc[trainIndex, 'mfg_TFLOPS'] =   2 * (w0 * h0) * (kernel_size * kernel_size * channel) * kernel_num * \
                                                    batch_size * batches_per_step / sec_per_iter / 10**12

            results = chunk.loc[trainIndex, :].values.tolist()

            print(results)
            resultsFinal.append(results)

            chunkNum += 1

        filepathOut = f'results_{dtype}.csv'

        df = pd.DataFrame.from_records(resultsFinal)
        df.to_csv(filepathOut, index=False)

if __name__ == '__main__':
    headerLine = processRawData()
    calculateTFLOPS(headerLine)
