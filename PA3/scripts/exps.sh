# Learning rate experiments
nohup python scripts/main.py --arch models/1.json --lr 0.1 --batch_size 20 --init 1 --model_name 1.1 &
nohup python scripts/main.py --arch models/1.json --lr 0.01 --batch_size 20 --init 1 --model_name 1.2 &
nohup python scripts/main.py --arch models/1.json --lr 0.001 --batch_size 20 --init 1 --model_name 1.3 &
nohup python scripts/main.py --arch models/1.json --lr 0.0001 --batch_size 20 --init 1 --model_name 1.4 &

# Batch size experiments
nohup python scripts/main.py --arch models/1.json --lr 0.001 --batch_size 20 --init 1 --model_name 1.5 &
nohup python scripts/main.py --arch models/1.json --lr 0.001 --batch_size 50 --init 1 --model_name 1.6 &
nohup python scripts/main.py --arch models/1.json --lr 0.001 --batch_size 100 --init 1 --model_name 1.7 &
nohup python scripts/main.py --arch models/1.json --lr 0.001 --batch_size 150 --init 1 --model_name 1.8 &

# Initializer experiment
nohup python scripts/main.py --arch models/1.json --lr 0.001 --batch_size 50 --init 1 --model_name 1.9 &
nohup python scripts/main.py --arch models/1.json --lr 0.001 --batch_size 50 --init 2 --model_name 1.10 &

