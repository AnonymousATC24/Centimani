# Allreduce configurations
Each chip has 64500000 single floating point numbers, ~250 MBytes per graphcore chip

# Login to GraphCloud

# Set up GraphCloud virtual environment
source /opt/gc/poplar_sdk-ubuntu_18_04-2.4.0+856-d16ca54529/poplar-ubuntu_18_04-2.4.0+2529-969064e2df/enable.sh

source /opt/gc/poplar_sdk-ubuntu_18_04-2.4.0+856-d16ca54529/popart-ubuntu_18_04-2.4.0+2529-969064e2df/enable.sh

source ~/workspace/poptorch_env/bin/activate

# Compile and test bandwidth (GB/s)
make;

./gcl_allreduce_example numberofDevices(such as 2, 4, 8, and 16);

# Bandwidth (GB/s) on NVIDIA A100 GPUs
<img src="GC.PNG" alt="A100" width="600"/>
