# FERD
The official code for FERD: Fairness-Enhanced Data-Free Robustness Distillation

## Install
```bash
pip install -r requirements.txt
```


## Demo
```bash
python test.py \
    --teacher_model WRN_34_10 \
    --student_model ResNet18 \
    --data CIFAR10 \
    --batch_size 256 \
    --epoch 220 \
    --N_G 200 --N_S 400 \
    --lr 0.1 \
    --lr_G 2e-3 \
    --lr_z 0.015 \
    --gen_dim_z 100 \
    --warmup 20 \
    --adv 1 --bn 5 --oh 1 --uni 5 \
    --int_pop 3 \
    --experiment_name c10_wrn10_res18
```
