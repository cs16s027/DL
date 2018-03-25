# Note: all paths referenced here are relative to the Docker container.
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
source /tools/config.sh
source activate py27
cd /storage/home/karthikt/DL/PA3

python -u scripts/train.py --arch models/1.json --lr 0.001 --batch_size 50 --init 1 --model_name 1.sub
python -u scripts/test.py --arch models/1.json --lr 0.001 --batch_size 50 --init 1 --model_name 1.sub
