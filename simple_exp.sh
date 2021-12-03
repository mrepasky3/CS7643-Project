#!/bin/sh

python run_experiments.py --data_source "Toy" --experiment_type "times" --task "sum" --max_size 20 --epochs 100
python run_experiments.py --data_source "Toy" --experiment_type "times" --task "max" --max_size 20 --epochs 100
python run_experiments.py --data_source "Toy" --experiment_type "times" --task "range" --max_size 20 --epochs 100
python run_experiments.py --data_source "Toy" --experiment_type "times" --task "mode" --max_size 20 --epochs 100
python run_experiments.py --data_source "Toy" --experiment_type "times" --task "product" --max_size 20 --epochs 100


python run_experiments.py --data_source "MNIST" --experiment_type "times" --task "sum" --max_size 20 --epochs 100
python run_experiments.py --data_source "MNIST" --experiment_type "times" --task "max" --max_size 20 --epochs 100
python run_experiments.py --data_source "MNIST" --experiment_type "times" --task "range" --max_size 20 --epochs 100
python run_experiments.py --data_source "MNIST" --experiment_type "times" --task "mode" --max_size 20 --epochs 100
python run_experiments.py --data_source "MNIST" --experiment_type "times" --task "product" --max_size 20 --epochs 100


python run_experiments.py --data_source "OMNI" --experiment_type "times" --task "unique" --max_size 30 --epochs 100 --batch_size 50
