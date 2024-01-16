#
system = 'Graphcore'

for dtype in ['fp16', 'fp32']:

    filepathIn = f'results_raw_{dtype}.txt'
    filepathOut = f'results_{dtype}.csv'

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
                    line = line.rstrip()

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

                    fpOut.write(line)
                    line = fpIn.readline()
                    continue

                #0.633    sec/itr.   647284.987433 items/sec,  74.829707 TFLOPS
                if line.find('sec/itr.'):
                    line = line.replace(' items/sec,  ', ',')
                    line = line.replace(' TFLOPS', ',')
                    line = line.replace(' ', '')
                    line = line.replace('sec/itr.', ',')
                    line = line.rstrip()


                fpOut.write(line)
                print(f"Line: {line}")


                line = fpIn.readline()