#!/bin/bash
#
#SBATCH --job-name=c10_wrn10_res18_w1510
#SBATCH --output=output/c10_wrn10_res18_w1510.txt
#SBATCH --error=error/c10_wrn10_res18_w1510.txt
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --nodelist=aias-compute-3
#SBATCH --qos=a800
#SBATCH --partition=a800

# CIFAR10

srun python test.py --teacher_model WRN_34_10 --student_model ResNet18 --data CIFAR10 --batch_size 256 --epoch 220 --N_G 200 --N_S 400 --lr 0.1 --lr_G 2e-3 --lr_z 0.015 --gen_dim_z 100 --warmup 20 --adv 1 --bn 5 --oh 1 --uni 5 --experiment_name c10_wrn10_res18_w1510

# CIFAR100

#srun python test.py --teacher_model WRN_34_10 --student_model ResNet18 --data CIFAR100 --batch_size 256 --epoch 220 --N_G 200 --N_S 400 --lr 0.1 --lr_G 2e-3 --lr_z 0.015 --gen_dim_z 100 --warmup 20 --experiment_name c100_wrn10_res18  54.20

#srun python test.py --teacher_model WRN_34_10 --student_model ResNet18 --data CIFAR100 --batch_size 256 --epoch 220 --N_G 400 --N_S 400 --lr 0.1 --lr_G 1e-3 --lr_z 0.015 --gen_dim_z 100 --warmup 20 --experiment_name c100_wrn10_res18  54.06

#srun python test.py --teacher_model WRN_34_10 --student_model ResNet18 --data CIFAR100 --batch_size 512 --epoch 220 --N_G 200 --N_S 400 --lr 0.2 --lr_G 2e-3 --lr_z 0.015 --gen_dim_z 100 --warmup 20 --experiment_name c100_wrn10_res18 56.24

#srun python test.py --teacher_model WRN_34_10 --student_model ResNet18 --data CIFAR100 --batch_size 512 --epoch 320 --N_G 200 --N_S 400 --lr 0.2 --lr_G 2e-3 --lr_z 0.015 --gen_dim_z 100 --warmup 20 --adv 1 --bn 5 --oh 1 --uni 5 --experiment_name c100_wrn10_res18_e320  


# Tiny_ImageNet
#srun python test.py --teacher_model PreActResNet34 --student_model PreActResNet18 --data tiny --batch_size 512 --epoch 220 --N_G 2 --N_S 2 --lr 0.2 --lr_G 2e-3 --lr_z 0.015 --gen_dim_z 100 --warmup 0 --adv 1 --bn 5 --oh 1 --uni 5 --experiment_name tiny_par34_par18_