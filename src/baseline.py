import datetime
import argparse
from base_models import load_model
from fairness import FREQUENCY
from myutils.io_helper import mem_usage, Logger, make_file_path
from myutils.data_processor import load_db_by_name
from myutils.data_generator import load_synthetic_data
from myutils.data_utils import SummaryWritter
from myutils.np_writter import Writter
import torch
import numpy as np
import pickle
import json
import random
from itertools import product
import os
from collections import OrderedDict

# Tune-able parameter
ALPHA = 0.
BETA = 0.1
NET_DEPTH = 3
SEED = 42
TEST_SIZE = 0.4
NUM_TEMPERS = 5
TOLERANCE = 1e-5
NUM_EPOCHS = 1000
BATCH_SIZE = 8192
PERIOD = 50
DECAY_FACTOR = 0.1
STEP_SIZE = 0.1
WEIGHT_DECAY = 0.1
MOMENTUM = 0.
TEMPERATURE = 1e0
FREQUENCY = 5
PENALTY = 0.0001
# Not tune-able parameter
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_ALG = "RWM"
DEFAULT_ML = "Intra"
DEFAULT_OBJ = "BCE"
DEFAULT_NET = "mlp"
DEFAULT_DB = "adult"
DEFAULT_GP = "sex"
WEIGHTS_PATH = "weights"
LOGS_PATH = "logs"
RESULTS_PATH = "results"
FIGURES_PATH = "figs"
CONFIG_PATH = "config"

def train_n_test(args):
    """
        :param X:  numpy matrix of features with dimensions numsamples x numdims
        :param y:  numpy array of labels with length numsamples. Should be numeric (0/1 labels binary classification)
        :param numsteps:  number of rounds to run the game
        :param grouplabels:  numpy array of numsamples numbers in the range [0, numgroups) denoting groups membership
        :param group_names:  list of groups names in relation to underlying data (e.g. [male, female])
        :param data_name:  name of the dataset being used to make plots clear
        :param gamma: maximum allowed max groups error by convergence
        :param relaxed: denotes whether or not we are solving the relaxed version of the problem
        :param model_type:  sklearn model type e.g. LinearRegression, LogisticRegression, etc.
        :param error_type:  for classification only! e.g. Total, FP, FN
        :param extra_error_types: set of error types which we want to plot
        :param pop_error_type: error type to use on population e.g. Total for FP/FN
        :param convergence_threshold: converges (early) when max change in sampleweights < convergence_threshold
        :param penalty: Regularization penalty for logistic regression
        :param C: inverse of regularization strength
        :param logistic_solver: Which underlying solver to use for logistic regression
        :param fit_intercept: Whether or not we should fit an additional intercept
        :param random_split_seed: the random state to perform the train test split on
        :param display_plots: denotes if plots should be displayed
        :param show_legend: denotes if plots should have legends with groups names
        :param save_models: denotes if models should be saved in each round (needed to extract mixtures)
        :param save_plots: determines if plots should be saved to a file
        :param dirname: name of directory to save plots/models in, if applicable (sub directory of s3 bucket, if applicable)
        :param test_size: if nonzero, proportion of data to be reserved for validation of training data
        :param max_logi_iters: max number of logistic regression iterations
        :param tol: tolerance of convergence for logistic regression
        :param lr: learning rate of gradient descent for MLP
        :param n_epochs: number of epochs per individual MLP model
        :param hidden_sizes: list of sizes for hidden layers of MLP - fractions (and 1) treated as proportions of numdims
        """
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # define method name
    base_name = args.obj_name + "Learner"
    postfix = ""
    if "pAUC" in args.obj_name:
        postfix += "_" + str(args.alpha) + "_" + str(args.beta)
    # load config
    config_path = make_file_path(args.config_folder, *(args.db_name, args.gp_name, args.net_name, base_name))
    config_path += postfix + ".json"
    if args.default:
        if os.path.exists(config_path):
            # flush the original namespace
            with open(config_path) as f:
                config_dict = json.load(f)
            sorted_config_dict = OrderedDict(sorted(config_dict.items()))
            args_dict = vars(args)
            args_dict.update(sorted_config_dict) # also update the args
    else:
        args_dict = vars(args)
        config_dict = {}
        for key, values in args_dict.items():
            if (isinstance(values, int) and not isinstance(values, bool)) or isinstance(values, float):
                config_dict[key] = values
        sorted_config_dict = OrderedDict(sorted(config_dict.items()))
        if not os.path.exists(config_path):
            with open(config_path, "w", encoding="utf8") as f:
                json.dump(sorted_config_dict, f, indent=2)

    # load the baseline config dict for loading baseline model later
    bce_config_path = make_file_path(args.config_folder, *(args.db_name, args.gp_name, args.net_name, "BCELearner"))
    bce_config_path += postfix + ".json"
    if os.path.exists(bce_config_path):
        # flush the original namespace
        with open(bce_config_path) as file:
            bce_config_dict = json.load(file)
        sorted_bce_config_dict = OrderedDict(sorted(bce_config_dict.items()))



    # create path for later
    logs_path = make_file_path(args.logs_folder, *(args.db_name, args.gp_name, args.net_name, base_name))
    logs_path = make_file_path(logs_path, **sorted_config_dict)
    logger = Logger(logs_path)

    load_path = make_file_path(args.weights_folder, *(args.db_name, args.gp_name, args.net_name, "BCELearner"))
    load_path = make_file_path(load_path, **sorted_bce_config_dict)
    load_path += postfix + ".pt"
    save_path = make_file_path(args.weights_folder, *(args.db_name, args.gp_name, args.net_name, base_name))
    save_path = make_file_path(save_path, **sorted_config_dict)
    save_path += postfix + ".pt"
    
    results_path = make_file_path(args.results_folder, *(args.db_name, args.gp_name, args.net_name, base_name))
    results_path = make_file_path(results_path, **sorted_config_dict)
    results_path += postfix + ".p"

    logger.log("Start fitting... - {}".format(datetime.datetime.now()), verbose=True)
    logger.log("Data: {} Group: {} Net: {} Method: {}".format(args.db_name, args.gp_name, args.net_name, base_name), verbose=True)

    # record loading time
    time_1 = datetime.datetime.now()

    logger.log("Reading... - mem {} Mb - {}".format(mem_usage(), datetime.datetime.now()), verbose=True)

    if args.db_name == 'synthetic':
        (train_data, train_writter), (test_data, test_writter) = load_synthetic_data(args.gp_name, ratio=args.test_size, seed=args.seed)
        # train_writter = SummaryWritter(X_train, Y_train, Z_train)
        # test_writter = SummaryWritter(X_test, Y_test, Z_test)
        # train_data = train_writter.grouplabelsplitter(X_train, Y_train, Z_train)
        # test_data = test_writter.grouplabelsplitter(X_test, Y_test, Z_test)
    else:
        (train_data, train_writter), (test_data, test_writter) = load_db_by_name(args.db_name, args.gp_name, ratio=args.test_size, seed=args.seed)

    # Initialze stat and result writter
    train_recorder = Writter()
    test_recorder = Writter()
    # sampleweights = np.ones(n_samples)  # initialize sample weights array to uniform
    # for g in range(0, n_groups):
    #     weight_per_sample = groupweights[0, g]
    #     sampleweights[groupindices[g]] = weight_per_sample
    # sampleweights = None

    logger.log("Number of features: {} Size of the train set: {} Size of the test set: {} ".format(train_writter.n_features, train_writter.n_samples, test_writter.n_samples), verbose=True, newline=False)
    for g in range(train_writter.n_groups):
        logger.log('Group: {} negative: {} positive: {} '.format(g, train_writter.n_grouplabel[2*g], train_writter.n_grouplabel[2*g+1]), newline=False, verbose=True)
    logger.log('')
    logger.log("Training... - mem {} Mb- {}".format(mem_usage(), datetime.datetime.now()), verbose=True)

    # record training time
    time_2 = datetime.datetime.now()

    # List for storing the model produced at every round if applicable
    Model = load_model(base_name)
    # Learner best responds to current weight by training a model on weighted sample points
    if args.net_name == 'mlp':
        modelhat = Model([train_writter.n_features] * args.net_depth, logger, beta=args.beta, lr=args.step_size, temp=args.temperature,
                         momentum=args.momentum, weight_decay=args.weight_decay, batch_size=args.batch_size,
                         period=args.period, multiplicative_factor=args.decay_factor, n_epochs=args.num_epochs,
                         num_tempers=args.num_tempers, tolerance=args.tolerance, verbose=args.verbose, device=args.device,
                         save_path=save_path, load_path=load_path)
    else:
        modelhat = Model(train_writter.n_features, logger, beta=args.beta, lr=args.step_size, temp=args.temperature,
                         momentum=args.momentum, weight_decay=args.weight_decay, batch_size=args.batch_size,
                         period=args.period, multiplicative_factor=args.decay_factor, n_epochs=args.num_epochs,
                         num_tempers=args.num_tempers, tolerance=args.tolerance, verbose=args.verbose, device=args.device,
                         save_path=save_path, load_path=load_path)

    train_summary, val_summary = modelhat.fit(train_data, train_writter, test_data, test_writter)

    time_3 = datetime.datetime.now()

    # read the record and update
    train_recorder.torch2np(train_summary)
    test_recorder.torch2np(val_summary)

    logger.log('---------- Training results ----------', verbose=args.verbose)
    logger.log('| MIS:{:.4f} AUC:{:.4f} pAUC:{:.4f} |'.format(train_recorder.mis_errors[-1],
                                                         train_recorder.auc_errors[-1],
                                                         train_recorder.pauc_errors[-1]), newline=False, verbose=args.verbose)

    for (i, j) in product(range(train_writter.n_groups), range(train_writter.n_groups)):
        logger.log("N-{} P-{} AUC:{:.4f} pAUC:{:.4f} |".format(i, j, train_recorder.auc_grouperrs[
            j + i * train_writter.n_groups, -1], train_recorder.pauc_grouperrs[
                                                              j + i * train_writter.n_groups, -1]), newline=False, verbose=args.verbose)
    logger.log("", verbose=args.verbose)
    logger.log('---------- Test results ----------', verbose=args.verbose)
    logger.log('| MIS:{:.4f} AUC:{:.4f} pAUC:{:.4f} |'.format(test_recorder.mis_errors[-1],
                                                         test_recorder.auc_errors[-1],
                                                         test_recorder.pauc_errors[-1]), newline=False, verbose=args.verbose)

    for (i, j) in product(range(test_writter.n_groups), range(test_writter.n_groups)):
        logger.log("N-{} P-{} AUC:{:.4f} pAUC:{:.4f} |".format(i, j, test_recorder.auc_grouperrs[
            j + i * test_writter.n_groups, -1], test_recorder.pauc_grouperrs[
                                                              j + i * test_writter.n_groups, -1]), newline=False, verbose=args.verbose)

    logger.log("", verbose=args.verbose)

    with open(results_path, "wb") as f:
        pickle.dump([train_writter, train_recorder, test_writter, test_recorder], f)

    time_4 = datetime.datetime.now()
    logger.log("Done! time_to_load: {} time_to_fit: {} time_to_save: {}".format(time_2 - time_1, time_3 - time_2,
                                                                                time_4 - time_3), verbose=True)

    return train_writter, train_recorder, test_writter, test_recorder

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
    parser.add_argument("--alg_name", type=str, default=DEFAULT_ALG, choices=['baseline', 'RWM', 'LSE'],
                        help="Methods to run.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_ML, choices=['Intra', 'Inter', 'All'], help="Model of fairness.")
    parser.add_argument("--net_name", type=str, default=DEFAULT_NET, choices=['linear', 'mlp'], help="Nets to run.")
    parser.add_argument("--db_name", type=str, default=DEFAULT_DB, choices=['adult', 'bank', 'communities', 'compas',  'default', 'german', 'mimiciii', 'synthetic'],
                        help="Database names.")
    parser.add_argument("--gp_name", type=str, default=DEFAULT_GP, choices=['sex', 'race', 'age', 'gaussian'],
                        help="Sensitive group names.")
    parser.add_argument("--param_files", type=str, default=None, nargs='+', help="Loads a file with many parameters.")
    parser.add_argument("--load_trained_model", type=str, default=None, help="Loads a previously trained model.")
    parser.add_argument("--device", type=str, default=DEVICE, help="training on cpu or cuda")
    parser.add_argument("--verbose", action='store_true', help="print on the screen")
    parser.add_argument("--default", action='store_true', help='load default json file')
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
    parser.add_argument("--frequency", type=int, default=FREQUENCY)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--penalty", type=float, default=PENALTY)
    parser.add_argument("--decay_factor", type=float, default=DECAY_FACTOR)
    parser.add_argument("--alpha", type=float, default=ALPHA, help="pAUC lower bound")
    parser.add_argument("--beta", type=float, default=BETA, help="pAUC upper bound")

    args = parser.parse_args()
    train_n_test(args)

    return

if __name__ == "__main__":

    main()