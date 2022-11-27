#!/bin/bash
cd src
db='adult'
gp='sex'
echo "-------------------------------------------------------------------------"
echo "--------------------------- Dataset = ${db} -----------------------------"
echo "-------------------------------------------------------------------------"
for i in {38..42..2}
do
    echo "----------------------------  SEED = ${i}  ------------------------------"
    echo "-------------------------------------------------------------------------"
    python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --name baseline --new_seed ${i}
    python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --default
    echo "-------------------------------------------------------------------------"
    python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --name baseline --new_seed ${i}
    python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --default
    # echo "-------------------------------------------------------------------------"
    # python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --name fairness --new_seed ${i}
    # python fairness.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --default
done
db='bank'
gp='age'
echo "-------------------------------------------------------------------------"
echo "--------------------------- Dataset = ${db} -----------------------------"
echo "-------------------------------------------------------------------------"
for i in {38..42..2}
do
    echo "----------------------------  SEED = ${i}  ------------------------------"
    echo "-------------------------------------------------------------------------"
    python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --name baseline --new_seed ${i}
    python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --default
    echo "-------------------------------------------------------------------------"
    python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --name baseline --new_seed ${i}
    python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --default
    # echo "-------------------------------------------------------------------------"
    # python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --name fairness --new_seed ${i}
    # python fairness.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --default
done
db='compas'
gp='race'
echo "-------------------------------------------------------------------------"
echo "--------------------------- Dataset = ${db} -----------------------------"
echo "-------------------------------------------------------------------------"
for i in {38..42..2}
do
    echo "----------------------------  SEED = ${i}  ------------------------------"
    echo "-------------------------------------------------------------------------"
    python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --name baseline --new_seed ${i}
    python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --default
    echo "-------------------------------------------------------------------------"
    python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --name baseline --new_seed ${i}
    python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --default
    # echo "-------------------------------------------------------------------------"
    # python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --name fairness --new_seed ${i}
    # python fairness.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --default
done
db='default'
gp='sex'
echo "-------------------------------------------------------------------------"
echo "--------------------------- Dataset = ${db} -----------------------------"
echo "-------------------------------------------------------------------------"
for i in {38..42..2}
do
    echo "----------------------------  SEED = ${i}  ------------------------------"
    echo "-------------------------------------------------------------------------"
    python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --name baseline --new_seed ${i}
    python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --default
    echo "-------------------------------------------------------------------------"
    python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --name baseline --new_seed ${i}
    python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --default
    # echo "-------------------------------------------------------------------------"
    # python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --name fairness --new_seed ${i}
    # python fairness.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --default
done
db='adult'
gp='sex'
echo "-------------------------------------------------------------------------"
echo "--------------------------- Dataset = ${db} -----------------------------"
echo "-------------------------------------------------------------------------"
for i in {38..42..2}
do
    echo "----------------------------  SEED = ${i}  ------------------------------"
    # echo "-------------------------------------------------------------------------"
    # python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --name baseline --new_seed ${i}
    # python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --default
    # echo "-------------------------------------------------------------------------"
    # python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --name baseline --new_seed ${i}
    # python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --default
    echo "-------------------------------------------------------------------------"
    python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --name fairness --new_seed ${i}
    python fairness.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --default
done
python printresult.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC
db='bank'
gp='age'
echo "-------------------------------------------------------------------------"
echo "--------------------------- Dataset = ${db} -----------------------------"
echo "-------------------------------------------------------------------------"
for i in {38..42..2}
do
    echo "----------------------------  SEED = ${i}  ------------------------------"
    # echo "-------------------------------------------------------------------------"
    # python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --name baseline --new_seed ${i}
    # python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --default
    # echo "-------------------------------------------------------------------------"
    # python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --name baseline --new_seed ${i}
    # python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --default
    echo "-------------------------------------------------------------------------"
    python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --name fairness --new_seed ${i}
    python fairness.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --default
done
python printresult.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC
db='compas'
gp='race'
echo "-------------------------------------------------------------------------"
echo "--------------------------- Dataset = ${db} -----------------------------"
echo "-------------------------------------------------------------------------"
for i in {38..42..2}
do
    echo "----------------------------  SEED = ${i}  ------------------------------"
    # echo "-------------------------------------------------------------------------"
    # python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --name baseline --new_seed ${i}
    # python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --default
    # echo "-------------------------------------------------------------------------"
    # python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --name baseline --new_seed ${i}
    # python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --default
    echo "-------------------------------------------------------------------------"
    python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --name fairness --new_seed ${i}
    python fairness.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --default
done
python printresult.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC
db='default'
gp='sex'
echo "-------------------------------------------------------------------------"
echo "--------------------------- Dataset = ${db} -----------------------------"
echo "-------------------------------------------------------------------------"
for i in {38..42..2}
do
    echo "----------------------------  SEED = ${i}  ------------------------------"
    # echo "-------------------------------------------------------------------------"
    # python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --name baseline --new_seed ${i}
    # python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name BCE --default
    # echo "-------------------------------------------------------------------------"
    # python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --name baseline --new_seed ${i}
    # python baseline.py --db_name ${db} --gp_name ${gp} --net_name mlp --obj_name AUC --default
    echo "-------------------------------------------------------------------------"
    python jsonscript.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --name fairness --new_seed ${i}
    python fairness.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC --default
done
python printresult.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC
