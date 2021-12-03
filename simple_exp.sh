#!/bin/sh

#python run_experiments.py --data_source "Toy" --experiment_type "simple" --task "sum" --max_size 20 --epochs 100
#python run_experiments.py --data_source "Toy" --experiment_type "simple" --task "max" --max_size 20 --epochs 100
#python run_experiments.py --data_source "Toy" --experiment_type "simple" --task "range" --max_size 20 --epochs 100
#python run_experiments.py --data_source "Toy" --experiment_type "simple" --task "mode" --max_size 20 --epochs 100
#python run_experiments.py --data_source "Toy" --experiment_type "simple" --task "product" --max_size 20 --epochs 100


#python run_experiments.py --data_source "MNIST" --experiment_type "simple" --task "sum" --max_size 20 --epochs 100
#python run_experiments.py --data_source "MNIST" --experiment_type "simple" --task "max" --max_size 20 --epochs 100
#python run_experiments.py --data_source "MNIST" --experiment_type "simple" --task "range" --max_size 20 --epochs 100
#python run_experiments.py --data_source "MNIST" --experiment_type "simple" --task "mode" --max_size 20 --epochs 100
#python run_experiments.py --data_source "MNIST" --experiment_type "simple" --task "product" --max_size 20 --epochs 100


python run_experiments.py --data_source "OMNI" --experiment_type "simple" --task "unique" --max_size 30 --epochs 100 --batch_size 50
