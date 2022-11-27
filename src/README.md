# pAUC-fairness

Run default baseline

```
python baseline.py --db_name adult --gp_name sex --net_name mlp --obj_name BCE --default True
```

Run default fairness

```
python fairness.py --db_name adult --gp_name sex --net_name mlp --alg_name RWM --model_name Intra --obj_name BCE --default True
```

Run baseline

```
python baseline.py --db_name adult --gp_name sex --net_name mlp --obj_name BCE --alpha 0.0 --beta 0.1 --net_depth 3 --seed 42 --validation_size 0.4 --num_tempers 5 --tolerance 1e-4 --num_epochs 2000 --batch_size 8192 --period 30 --multiplicative_factor 1e-1 --learning_rate 1e-1 --weight_decay 1e-3 --momentum 0.0 --learning_ratio 1e1 --regularization 1e-3
```

Run fairness

```
python fairness.py --db_name adult --gp_name sex --net_name mlp --alg_name RWM --model_name Intra --obj_name BCE --alpha 0.0 --beta 0.1 --net_depth 3 --seed 42 --validation_size 0.4 --num_tempers 5 --tolerance 1e-4 --num_epochs 2000 --batch_size 8192 --period 30 --multiplicative_factor 1e-1 --learning_rate 1e-1 --weight_decay 1e-3 --momentum 0.0 --learning_ratio 1e1 --regularization 1e-3
```