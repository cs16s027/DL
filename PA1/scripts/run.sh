python scripts/train.py --lr 1.0 --momentum 0.5 --num_hidden 4 --sizes 300,200,100,100 --activation sigmoid --loss ce --opt gd --batch_size 50 --anneal False --save_dir models/ --expt_dir logs/ --train data/train.csv --val data/val.csv --test data/test.csv
