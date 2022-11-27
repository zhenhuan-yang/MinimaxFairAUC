import os
import argparse
import numpy as np
import json
import pickle
import sys
from collections import OrderedDict
from baseline import train_n_test
from fairness import minimax_fair
from myutils.roc_plotter import Plotter
from myutils.io_helper import make_file_path

# Tune-able parameter
ALPHA = 0.0
BETA = 0.1
NET_DEPTH = 1
SEED = 42
TEST_SIZE = 0.4
NUM_TEMPERS = 5
TOLERANCE = 1e-5
NUM_EPOCHS = 1000
BATCH_SIZE = 256
PERIOD = 50
DECAY_FACTOR = 0.1
STEP_SIZE = .1
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.
TEMPERATURE = 5.
PENALTY = 0.0001
FREQUENCY = 5
# Not tune-able parameter
DEVICE = "cpu"
DEFAULT_ALG = "RWM"
DEFAULT_ML = "Intra"
DEFAULT_OBJ = "BCE"
DEFAULT_NET = "mlp"
DEFAULT_DB = "adult"
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
    parser.add_argument("--alg_name", type=str, default=DEFAULT_ALG, choices=['baseline', 'RWM', 'LSE', 'GDA'],
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
    parser.add_argument("--verbose", type=bool, default=VERBOSE, help="logger.log on the screen")
    parser.add_argument("--default", type=bool, default=DEFAULT, help='load default json file')
    parser.add_argument("--seed", type=int, default=SEED, help="Random state")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--period", type=int, default=PERIOD)
    parser.add_argument("--num_tempers", type=int, default=NUM_TEMPERS)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--net_depth", type=int, default=NET_DEPTH)
    parser.add_argument("--step_size", type=float, default=STEP_SIZE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--lr_type", type=str, default=RATE_TYPE)
    parser.add_argument("--frequency", type=int, default=FREQUENCY)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--penalty", type=float, default=PENALTY)
    parser.add_argument("--decay_factor", type=float, default=DECAY_FACTOR)
    parser.add_argument("--alpha", type=float, default=ALPHA, help="pAUC lower bound")
    parser.add_argument("--beta", type=float, default=BETA, help="pAUC upper bound")


    args = parser.parse_args()

    seed_list = [38, 40, 42]
    baseline_agg_auc_grouperrs = np.zeros((4, len(seed_list)))
    baseline_agg_auc_errors = np.zeros(len(seed_list))
    agg_auc_grouperrs = np.zeros((4, len(seed_list)))
    agg_auc_errors = np.zeros(len(seed_list))

    # define method name
    fair_name = args.alg_name + args.model_name + args.obj_name + "Learner"
    base_name = args.obj_name + "Learner"
    postfix = ""
    if "pAUC" in args.obj_name:
        postfix += "_" + str(args.alpha) + "_" + str(args.beta)
    # load config
    config_path = make_file_path(args.config_folder, *(args.db_name, args.gp_name, args.net_name, base_name))
    config_path += postfix + ".json"

    for idx, seed in enumerate(seed_list):

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

        # update only seed
        
        args_dict = vars(args)
        args_dict.update(sorted_config_dict)
        args.seed = seed
        
        # dump dict
        config_dict = {}
        for key, values in args_dict.items():
            if (isinstance(values, int) and not isinstance(values, bool)) or isinstance(values, float):
                config_dict[key] = values
        sorted_config_dict = OrderedDict(sorted(config_dict.items()))
        with open(config_path, "w", encoding="utf8") as f:
            json.dump(sorted_config_dict, f, indent=2)

        base_load_path = make_file_path(args.results_folder, *(args.db_name, args.gp_name, args.net_name, base_name))
        base_load_path = make_file_path(base_load_path, **sorted_config_dict)
        base_load_path += postfix + ".p"
        with open(base_load_path, "rb") as f:
            baseline_train_writter, baseline_train_recorder, baseline_test_writter, baseline_test_recorder = pickle.load(f)

        # define method name
        fair_name = args.alg_name + args.model_name + args.obj_name + "Learner"
        base_name = args.obj_name + "Learner"
        postfix = ""
        if "pAUC" in args.obj_name:
            postfix += "_" + str(args.alpha) + "_" + str(args.beta)
        # load config
        config_path = make_file_path(args.config_folder, *(args.db_name, args.gp_name, args.net_name, args.lr_type, fair_name))
        config_path += postfix + ".json"
        with open(config_path) as f:
                config_dict = json.load(f)
        sorted_config_dict = OrderedDict(sorted(config_dict.items()))

        # update seed
        args_dict = vars(args)
        args_dict.update(sorted_config_dict)
        args.seed = seed

        # dump dict
        config_dict = {}
        for key, values in args_dict.items():
            if (isinstance(values, int) and not isinstance(values, bool)) or isinstance(values, float):
                config_dict[key] = values
        sorted_config_dict = OrderedDict(sorted(config_dict.items()))
        with open(config_path, "w", encoding="utf8") as f:
            json.dump(sorted_config_dict, f, indent=2)

        results_path = make_file_path(args.results_folder, *(args.db_name, args.gp_name, args.net_name, args.lr_type, fair_name))
        results_path = make_file_path(results_path, **sorted_config_dict)
        results_path += postfix + ".p"
        
        with open(results_path, "rb") as f:
            train_writter, train_recorder, test_writter, test_recorder = pickle.load(f)

        baseline_agg_auc_grouperrs[:, idx] = baseline_test_recorder.agg_auc_grouperrs[:, -1]
        baseline_agg_auc_errors[idx] = baseline_test_recorder.agg_auc_errors[-1]

        agg_auc_grouperrs[:, idx] = test_recorder.agg_auc_grouperrs[:, -1]
        agg_auc_errors[idx] = test_recorder.agg_auc_errors[-1]

    file_name = 'table2'
    root = os.path.dirname(os.getcwd())
    file_path = os.path.join(root, file_name)

    with open(file_path, 'a') as f:
        f.write('Dataset: {}'.format(args.db_name))
        f.write('\n')
        f.write('Baseline | Overall: ')
        f.write(str(baseline_agg_auc_errors.mean()))
        f.write(' +/- ')
        f.write(str(baseline_agg_auc_errors.std()))
        f.write(' Min/Max: ')
        f.write(str(np.mean(baseline_agg_auc_grouperrs.min(axis=0) / baseline_agg_auc_grouperrs.max(axis=0))))
        f.write(' +/- ')
        f.write(str(np.std(baseline_agg_auc_grouperrs.min(axis=0) / baseline_agg_auc_grouperrs.max(axis=0))))
        f.write('\n')
        f.write('Fairness | Overall: ')
        f.write(str(agg_auc_errors.mean()))
        f.write(' +/- ')
        f.write(str(agg_auc_errors.std()))
        f.write(' Min/Max: ')
        f.write(str(np.mean(agg_auc_grouperrs.min(axis=0) / agg_auc_grouperrs.max(axis=0))))
        f.write(' +/- ')
        f.write(str(np.std(agg_auc_grouperrs.min(axis=0) / agg_auc_grouperrs.max(axis=0))))
        f.write('\n')

if __name__ == "__main__":

    main()

