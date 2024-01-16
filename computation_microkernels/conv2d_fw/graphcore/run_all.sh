source /opt/gc/poplar_sdk-ubuntu_18_04-2.4.0+856-d16ca54529/poplar-ubuntu_18_04-2.4.0+2529-969064e2df/enable.sh
source /opt/gc/poplar_sdk-ubuntu_18_04-2.4.0+856-d16ca54529/popart-ubuntu_18_04-2.4.0+2529-969064e2df/enable.sh
popc --version
export IPUOF_CONFIG_PATH=~/.ipuof.conf.d/lr72-3-9-16ipu.conf
export TF_POPLAR_FLAGS=--executable_cache_path=/localdata/$USER/tmp
export POPTORCH_CACHE_DIR=/localdata/brucew/tmp
source ~/workspace/poptorch_env/bin/activate

echo "source run_loop_infer_fp16.sh"
echo "source run_loop_infer_fp32.sh"
