import os
import argparse
import numpy as np
import json
from collections import OrderedDict
from myutils.io_helper import make_file_path

# Tune-able parameter

NEW_SEED = 40
NAME = 'baseline'

# Not tune-able parameter
DEVICE = "cpu"
DEFAULT_ALG = "RWM"
DEFAULT_ML = "All"
DEFAULT_OBJ = "AUC"
DEFAULT_NET = "mlp"
DEFAULT_DB = "german"
DEFAULT_GP = "sex"
RATE_TYPE = 'const'
WEIGHTS_PATH = "weights"
LOGS_PATH = "logs"
RESULTS_PATH = "results"
FIGURES_PATH = "figs"
CONFIG_PATH = "config"
VERBOSE = True
DEFAULT = True

def main():
    parser = argparse.ArgumentParser(description="Fits the model.")
    parser.add_argument("--config_folder", type=str, default=CONFIG_PATH,
                        help="Load hyper parameter.")
    parser.add_argument("--weights_folder", type=str, default=WEIGHTS_PATH,
                        help="Name of the experiment.")
    parser.add_argument("--logs_folder", type=str, default=LOGS_PATH,
                        help="Save experiment log.")
    parser.add_argument("--results_folder", type=str, default=RESULTS_PATH,
                        help="Save experiment result.")
    parser.add_argument("--figs_folder", type=str, default=FIGURES_PATH,
                        help="Save experiment figure.")
    parser.add_argument("--obj_name", type=str, default=DEFAULT_OBJ, choices=['AUC', 'pAUC', 'BCE', 'ThrespAUC', 'SingleThrespAUC'],
                        help="Objectives to run.")
    parser.add_argument("--alg_name", type=str, default=DEFAULT_ALG, choices=['baseline', 'RWM', 'LSE', 'GDA', 'LAG', 'REG'],
                        help="Methods to run.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_ML, choices=['Intra', 'Inter', 'All'], help="Model of fairness.")
    parser.add_argument("--net_name", type=str, default=DEFAULT_NET, choices=['linear', 'mlp'], help="Nets to run.")
    parser.add_argument("--db_name", type=str, default=DEFAULT_DB, choices=['adult', 'bank', 'communities', 'compas', 'default', 'german', 'mimiciii'],
                        help="Database names.")
    parser.add_argument("--gp_name", type=str, default=DEFAULT_GP, choices=['sex', 'race', 'age'],
                        help="Sensitive group names.")
    parser.add_argument("--param_files", type=str, default=None, nargs='+', help="Loads a file with many parameters.")
    parser.add_argument("--load_trained_model", type=str, default=None, help="Loads a previously trained model.")
    parser.add_argument("--device", type=str, default=DEVICE, help="training on cpu or cuda")
    parser.add_argument("--verbose", action='store_true', help="logger.log on the screen")
    parser.add_argument("--default", action='store_true', help='load default json file')
    parser.add_argument("--lr_type", type=str, default=RATE_TYPE)
    parser.add_argument("--new_seed", type=int, default=NEW_SEED)
    parser.add_argument("--name", type=str, default=NAME, choices=['baseline', 'fairness'])

    args = parser.parse_args()

    new_seed = args.new_seed + 0
    name = args.name + ''
    
    del args.new_seed
    del args.name

    if name == 'baseline':

        # define method name
        base_name = args.obj_name + "Learner"
        postfix = ""
        if "pAUC" in args.obj_name:
            postfix += "_" + str(args.alpha) + "_" + str(args.beta)
        # load config
        config_path = make_file_path(args.config_folder, *(args.db_name, args.gp_name, args.net_name, base_name))
        config_path += postfix + ".json"
        with open(config_path) as f:
            config_dict = json.load(f)
        sorted_config_dict = OrderedDict(sorted(config_dict.items()))

    # # update only seed
    # args_dict = vars(args)
    # args_dict.update(sorted_config_dict)
    # args.seed = args.new_seed

    # # dump config
    # config_dict = {}
    # for key, values in args_dict.items():
    #     if (isinstance(values, int) and not isinstance(values, bool)) or isinstance(values, float):
    #         config_dict[key] = values
    # sorted_config_dict = OrderedDict(sorted(config_dict.items()))
    # with open(config_path, "w", encoding="utf8") as f:
    #     json.dump(sorted_config_dict, f, indent=2)

    if name == 'fairness':

        # define method name
        fair_name = args.alg_name + args.model_name + args.obj_name + "Learner"
        postfix = ""
        if "pAUC" in args.obj_name:
            postfix += "_" + str(args.alpha) + "_" + str(args.beta)
        # load config
        config_path = make_file_path(args.config_folder, *(args.db_name, args.gp_name, args.net_name, args.lr_type, fair_name))
        config_path += postfix + ".json"
        with open(config_path) as f:
            config_dict = json.load(f)
        sorted_config_dict = OrderedDict(sorted(config_dict.items()))

    # update only seed
    args_dict = vars(args)
    args_dict.update(sorted_config_dict)
    args.seed = new_seed
    
    # dump dict
    config_dict = {}
    for key, values in args_dict.items():
        if (isinstance(values, int) and not isinstance(values, bool)) or isinstance(values, float):
            config_dict[key] = values
    sorted_config_dict = OrderedDict(sorted(config_dict.items()))
    with open(config_path, "w", encoding="utf8") as f:
        json.dump(sorted_config_dict, f, indent=2)

    


if __name__ == "__main__":

    main()

