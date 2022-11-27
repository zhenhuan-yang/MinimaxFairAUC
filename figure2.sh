#!/bin/bash
cd src
db='adult'
gp='sex'
echo "-------------------------------------------------------------------------"
echo "--------------------------- Dataset = ${db} -----------------------------"
echo "-------------------------------------------------------------------------"
python plotresult.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC
db='bank'
gp='age'
echo "-------------------------------------------------------------------------"
echo "--------------------------- Dataset = ${db} -----------------------------"
echo "-------------------------------------------------------------------------"
python plotresult.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC
db='compas'
gp='race'
echo "-------------------------------------------------------------------------"
echo "--------------------------- Dataset = ${db} -----------------------------"
echo "-------------------------------------------------------------------------"
python plotresult.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC
db='default'
gp='sex'
echo "-------------------------------------------------------------------------"
echo "--------------------------- Dataset = ${db} -----------------------------"
echo "-------------------------------------------------------------------------"
python plotresult.py --db_name ${db} --gp_name ${gp} --net_name mlp --alg_name RWM --model_name All --obj_name AUC
