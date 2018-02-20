# To be run from the submission folder
# Please place the data in the data folder, or enter the data path for these arguments
python scripts/train.py --lr 0.001 --momentum 0.5 --num_hidden 2 --sizes 300,300 --activation sigmoid --loss ce --opt adam --batch_size 20 --anneal true --save_dir Kaggle_subs/models --expt_dir Kaggle_subs/logs --train data/train.csv --val data/val.csv --test data/test.csv --pretrain false
