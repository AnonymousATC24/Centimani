# Graphcore

# Request POD Access

Contact support@alcf.anl.gov.

### Public Key

Give your public key to Graphcore.

## Connecting to Graphcore

Regular connection:

```bash
ssh __GC_POD_USERNAME__@38.83.162.168
```

## Create Workspace

```bash
mkdir /localdata/$USER/workspace
ln -s /localdata/$USER/workspace/ ~/workspace
```

```bash
source /opt/gc/poplar_sdk-ubuntu_18_04-2.4.0+856-d16ca54529/poplar-ubuntu_18_04-2.4.0+2529-969064e2df/enable.sh
source /opt/gc/poplar_sdk-ubuntu_18_04-2.4.0+856-d16ca54529/popart-ubuntu_18_04-2.4.0+2529-969064e2df/enable.sh
```

Check if poplar is setup correctly

```bash
popc --version
```

You should see this:

```bash
POPLAR version 2.4.0 (10a96ee536)
clang version 14.0.0 (ad697584ed527786ca41847bf56b2019532cd738)
```

Set the following flags. Prefer not to set the *LOG_LEVEL flags as it produces too many logs, unless you plan to debug at the popart / poplib/ poplar level.

All these commands can be put as part of your .bashrc file. 

```bash
export IPUOF_CONFIG_PATH=~/.ipuof.conf.d/lr72-3-9-16ipu.conf
export TF_POPLAR_FLAGS=--executable_cache_path=/localdata/$USER/tmp
export POPTORCH_CACHE_DIR=/localdata/$USER/tmp
#export POPART_LOG_LEVEL=INFO
#export POPLAR_LOG_LEVEL=INFO
#export POPLIBS_LOG_LEVEL=INFO
```


## PopArt Environment Setup

### Create PopArt Environment

This only needs to be done once.

'''bash
virtualenv -p python3 ~/workspace/poptorch_env
source ~/workspace/poptorch_env/bin/activate
pip3 install -U pip
pip install numpy
'''

## Clone

```bash
cd ~
git clone https://github.com/argonne-lcf/AIAcceleratorsSC22.git
```

## Run

```bash
source ~/AIAcceleratorsSC22/microkernels/relu/graphcore/run_all.sh
```

The above command will echo two 'source' commands to the screen.

- Run the first command that is for fp16.  
- Copy the screen output from all four runs. 
- Paste them into 'results_raw_fp16.txt' on your dev machine.
- Repeat process for fp32.
- python3 massage.py

## Compile Notes

A script will be compiled every time by default.

You may add the following commands to your .bashrc file to prevent the extra compiles.

```bash
echo 'export TF_POPLAR_FLAGS=--executable_cache_path=$HOME/tmp/'
export TF_POPLAR_FLAGS=--executable_cache_path=$HOME/tmp/

echo 'export POPTORCH_CACHE_DIR=$HOME/tmp/'
export POPTORCH_CACHE_DIR=$HOME/tmp/
```

## Handy Commands

```bash
gc-inventory |  grep "ipu utilisation:"
gc-info --list-devices
IPUOF_LOG_LEVEL=DEBUG gc-info -l
gc-monitor
```

## Profiling

### Download

```text
https://www.graphcore.ai/developer/popvision-tools#downloads
```

- Download both codes.
- Set permissions on both to executable.
- Execute app of choice.

### Environmental Variables:

```bash
POPLAR_ENGINE_OPTIONS = '{"autoReport.all":"true", "autoReport.directory":"./profile"}'
```