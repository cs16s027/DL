# Note: all paths referenced here are relative to the Docker container.
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
source /tools/config.sh
source activate py27
cd /storage/home/karthikt/DL/PA3

python -u scripts/train.py --arch exps/models/master_5conv_bn_2dropout.json --lr 0.0001 --batch_size 100 --init 1 --save_dir exps/models --expt_dir exps/logs --model_name master_5conv_bn_2dropout
