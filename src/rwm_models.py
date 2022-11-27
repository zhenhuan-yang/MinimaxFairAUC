import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from itertools import product
from itertools import permutations
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from myutils.model_helper import loss_helper, model_helper
from myutils.torch_writter import TorchWritter
from copy import deepcopy
from myutils.roc_optimizer import mis, pauc
from myutils.model_helper import AbsClassifier

class RWMBCELearner(AbsClassifier):
    '''
    '''
    def __init__(self, size, logger, beta=0.1, lr=0.001, momentum=0.9, weight_decay=0.0001, batch_size=1, temp=0.1, period=1,
                 multiplicative_factor=0.1, n_epochs=1000, frequency=1, regularization=0., tolerance=0.0001, num_tempers=3, lr_type='sqrt', 
                 verbose=True, device="cpu", save_path="", load_path="", no_ablation='Intra'):
        super(RWMBCELearner, self).__init__()
        self.logger = logger
        self.device = device
        self.save_path = save_path
        self.load_path = load_path
        self.model = model_helper(size).to(self.device)
        if self.load_path:
            self.model.load_state_dict(torch.load(self.load_path))
        self.bestmodel = model_helper(size).to(self.device)
        self.bestmodel.load_state_dict(self.model.state_dict())
        self.beta = beta
        self.lr = lr
        self.m = momentum
        self.wd = weight_decay
        self.bs = batch_size
        self.temp = temp
        self.p = period
        self.mf = multiplicative_factor
        self.n_epochs = n_epochs
        self.freq = frequency
        self.reg = regularization
        self.tol = tolerance
        self.num_tempers = num_tempers
        self.rtype = lr_type
        self.verbose = verbose

    def fit(self, train_data, train_writter, val_data, val_writter):
        '''
        '''

        # send everything to device
        X_train, y_train, group_neg_loaders, group_pos_loaders = self.np2torch(train_data, train_writter, get_loader=True)
        X_val, y_val, _, _ = self.np2torch(val_data, val_writter, get_loader=False)

        # initialize the group weights
        self.group_ratios = torch.zeros(train_writter.n_groups, dtype=torch.float32).to(self.device)
        for i in range(train_writter.n_groups):
            self.group_ratios[i] = train_writter.n_grouplabel[2*i] + train_writter.n_grouplabel[2*i+1]
        self.group_ratios = self.group_ratios / sum(self.group_ratios)
        self.group_weights = self.group_ratios + 0. # initialize the group_weights uniformly cross groups
        # self.group_weights = torch.ones(train_writter.n_groups, dtype=torch.float32).to(self.device) / train_writter.n_groups  # initialize the group_weights uniformly

        self.group_weights_rec = []
        # self.group_weights_rec.append(self.group_weights.detach().clone())

        self.criterion = loss_helper('bce')(k=None)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.m, weight_decay=self.wd)
        self.scheduler = self.stepsize_helper()
        
        train_recorder = TorchWritter(train_writter, beta=self.beta, n_tempers=self.num_tempers)
        val_recorder = TorchWritter(val_writter, beta=self.beta, n_tempers=self.num_tempers)


        bestValLoss = float("inf")
        temper = self.num_tempers
        bestepoch = 0

        # initiaze the record by baseline
        with torch.no_grad():
            self.bestmodel.eval()
            train_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_train)).squeeze(), min=1e-6,
                                                max=1 - 1e-6)
            val_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_val)).squeeze(), min=1e-6,
                                            max=1 - 1e-6)
            self.group_weights_rec.append(self.group_weights.detach().clone())
            train_recorder.update_stats(train_y_pred_proba, y_train)
            val_recorder.update_stats(val_y_pred_proba, y_val)

        freq_count = 0
        loss_freq = torch.zeros(train_writter.n_groups).to(self.device)
        # run
        for epoch in range(1, self.n_epochs):

            self.model.train()

            loss_train = torch.zeros(train_writter.n_groups).to(self.device)

            for batch_idx, (group_neg_loader, group_pos_loader) in enumerate(zip(zip(*group_neg_loaders), zip(*group_pos_loaders))):

                X_batch_list = []
                n_batch_neg = []
                n_batch_list = []
                
                # loop through groups
                for i in range(train_writter.n_groups):
                    # load the data from ith group
                    X_group_neg, = group_neg_loader[i]
                    X_group_pos, = group_pos_loader[i]
                    X_batch_list.append(X_group_neg)
                    X_batch_list.append(X_group_pos)
                    n_batch_neg.append(X_group_neg.shape[0])
                    n_batch_list.append(X_group_neg.shape[0] + X_group_pos.shape[0])

                # concatenate before parse by model and then split before calculate loss
                X_batch = torch.cat(X_batch_list)
                
                self.optimizer.zero_grad()
                yy_pred = self.model(X_batch).view(-1)
                yy_group = torch.split(yy_pred, n_batch_list)
                # now get the loss by looping groups
                losses = []
                for i in range(train_writter.n_groups):
                    yy_batch = torch.ones_like(yy_group[i]) # requires_grad=False
                    yy_batch[:n_batch_neg[i]] = 0.
                    loss_group = self.criterion(yy_group[i], yy_batch)

                    losses.append(loss_group)
                    
                # prepare loss for model parameter
                loss = torch.stack(losses)
                loss_batch = loss.dot(self.group_weights)
                # backward the loss
                loss_batch.backward()
                # update model parameter
                self.optimizer.step()

                # update group_weights
                with torch.no_grad():
                    loss_ = loss.detach().clone()
                    loss_train += loss_
                    loss_freq += loss_
                    if freq_count % self.freq == 0:
                        loss_freq = 1. / max(1, freq_count) * loss_freq
                        new_group_weights = self.group_weights * torch.exp(self.temp * self.scheduler.get_last_lr()[0] * loss_freq)
                        self.group_weights = new_group_weights / new_group_weights.norm(p=1)
                        # self.group_weights = torch.clamp(self.group_weights, min=1e-6, max=1 - 1e-6)
                        freq_count = 0
                        loss_freq = torch.zeros(train_writter.n_groups).to(self.device)

                self.scheduler.step()
                freq_count += 1

            # evaluate on the validation
            with torch.no_grad():
                self.model.eval()
                yy_val = self.model(X_val).view(-1)
                losses_val = []
                # loop through groups - no batch
                for i in range(val_writter.n_groups):
                    loss_group_val = self.criterion(yy_val[val_writter.group_ind[i]], y_val[val_writter.group_ind[i]])
                    losses_val.append(loss_group_val)
                # prepare loss for model parameter
                loss_val = torch.stack(losses_val)
                loss_val_weighted = loss_val.dot(self.group_weights)
                loss_val_max = max(losses_val)
                bestValLoss, bestepoch, temper = self.ckeck_n_save(loss_val_max, epoch, bestValLoss, bestepoch, temper,
                                                                   tol=self.tol, num_tempers=self.num_tempers)

                self.logger.log(
                    'Epoch:{} Train Loss:{:.6f} Val Loss:{:.6f} Val Max:{:.6f} Group Loss/Weight '.format(epoch, loss_batch, loss_val_weighted,
                                                                                        loss_val_max),
                    newline=False, verbose=self.verbose)
                for g in range(train_writter.n_groups):
                    self.logger.log('{}:{:.6f}/{:.6f} '.format(g, loss_val[g].item(), self.group_weights[g].item()), newline=False, verbose=self.verbose)
                self.logger.log("", verbose=self.verbose)

                # save the record
                self.bestmodel.eval()
                train_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_train)).squeeze(), min=1e-6, max=1-1e-6)
                val_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_val)).squeeze(), min=1e-6, max=1-1e-6)
                self.group_weights_rec.append(self.group_weights.detach().clone())
                train_recorder.update_stats(train_y_pred_proba, y_train)
                val_recorder.update_stats(val_y_pred_proba, y_val)

                if temper == 0:
                    self.logger.log('Early stopping at epoch {} and best epoch {}'.format(epoch, bestepoch), verbose=self.verbose)
                    break

                
        return train_recorder, val_recorder

class RWMAUCLearner(AbsClassifier):
    '''
    '''
    def __init__(self, size, logger, beta=0.1, lr=0.001, momentum=0.9, weight_decay=0.0001, batch_size=256, temp=0.1, period=1,
                 multiplicative_factor=0.1, n_epochs=1000, frequency=1, regularization=0., tolerance=0.0001, num_tempers=3, lr_type='sqrt', 
                 verbose=True, device="cpu", save_path="", load_path="", no_ablation='Intra'):
        super(RWMAUCLearner, self).__init__()
        self.logger = logger
        self.device = device
        self.save_path = save_path
        self.load_path = load_path
        self.model = model_helper(size).to(self.device)
        if self.load_path:
            self.model.load_state_dict(torch.load(self.load_path))
        self.bestmodel = model_helper(size).to(self.device)
        self.bestmodel.load_state_dict(self.model.state_dict())
        self.beta = beta
        self.lr = lr
        self.m = momentum
        self.wd = weight_decay
        self.bs = batch_size
        self.temp = temp
        self.p = period
        self.mf = multiplicative_factor
        self.n_epochs = n_epochs
        self.freq = frequency
        self.reg = regularization
        self.tol = tolerance
        self.num_tempers = num_tempers
        self.rtype = lr_type
        self.verbose = verbose
        self.nab = no_ablation

    def fit(self, train_data, train_writter, val_data, val_writter):
        '''
        '''

        # send everything to device
        X_train, y_train, group_neg_loaders, group_pos_loaders = self.np2torch(train_data, train_writter, get_loader=True)
        X_val, y_val, _, _ = self.np2torch(val_data, val_writter, get_loader=False)

        # initialize the group weights
        self.groupgroup_ratios = torch.zeros(train_writter.n_groups ** 2, dtype=torch.float32).to(self.device)
        self.groupgroup_masks = torch.ones(train_writter.n_groups ** 2, dtype=torch.float32).to(self.device)
        for idx, (i, j) in enumerate(product(range(train_writter.n_groups), range(train_writter.n_groups))):
            self.groupgroup_ratios[idx] = train_writter.n_grouplabel[2*i] * train_writter.n_grouplabel[2*j+1]
            if self.nab == 'Intra':
                if i != j:
                    self.groupgroup_ratios[idx] = 0.
                    self.groupgroup_masks[idx] = 0.
            elif self.nab == 'Inter': 
                if i == j:
                    self.groupgroup_ratios[idx] = 0.
                    self.groupgroup_masks[idx] = 0.
            else:
                pass
        self.groupgroup_ratios = self.groupgroup_ratios / sum(self.groupgroup_ratios)
        self.groupgroup_weights = self.groupgroup_ratios + 0. # initialize the group_weights by portion in the group
        # self.groupgroup_weights = torch.ones(train_writter.n_groups ** 2, dtype=torch.float32).to(self.device) / train_writter.n_groups ** 2  # initialize the group_weights uniformly cross groups 
        self.groupgroup_weights_rec = []
        # self.groupgroup_weights_rec[0] = self.groupgroup_weights.detach().clone()

        self.criterion = loss_helper('pauc')(k=None)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.m, weight_decay=self.wd)
        self.scheduler = self.stepsize_helper()

        train_recorder = TorchWritter(train_writter, beta=self.beta, n_tempers=self.num_tempers)
        val_recorder = TorchWritter(val_writter, beta=self.beta, n_tempers=self.num_tempers)

        bestValLoss = float("inf")
        temper = self.num_tempers
        bestepoch = 0

        # initiaze the record by baseline
        with torch.no_grad():
            self.bestmodel.eval()
            train_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_train)).squeeze(), min=1e-6, max=1-1e-6)
            val_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_val)).squeeze(), min=1e-6, max=1-1e-6)
            self.groupgroup_weights_rec.append(self.groupgroup_weights.detach().clone())
            train_recorder.update_stats(train_y_pred_proba, y_train)
            val_recorder.update_stats(val_y_pred_proba, y_val)

        freq_count = 0
        loss_freq = torch.zeros(train_writter.n_groups ** 2).to(self.device)
        # run
        for epoch in range(1, self.n_epochs + 1):

            self.model.train()

            loss_train = torch.zeros(train_writter.n_groups ** 2).to(self.device)

            for batch_idx, (group_neg_loader, group_pos_loader) in enumerate(zip(zip(*group_neg_loaders), zip(*group_pos_loaders))):

                X_batch_list = []
                n_batch_list = []
                # clear gradient
                self.optimizer.zero_grad()

                # loop through groups
                for (i, j) in product(range(train_writter.n_groups), range(train_writter.n_groups)):
                    # load the data from ith group and jth group
                    X_group_neg, = group_neg_loader[i]
                    X_group_pos, = group_pos_loader[j]
                    X_batch_list.append(X_group_neg)
                    X_batch_list.append(X_group_pos)
                    n_batch_list.append(X_group_neg.shape[0])
                    n_batch_list.append(X_group_pos.shape[0])

                X_batch = torch.cat(X_batch_list)
                yy_batch = self.model(X_batch).view(-1)
                yy_group = torch.split(yy_batch, n_batch_list)

                losses = []
                for idx, (i, j) in enumerate(product(range(train_writter.n_groups), range(train_writter.n_groups))):
                    loss_group = self.criterion(yy_group[2 * idx], yy_group[2 * idx + 1])
                    losses.append(loss_group)

                # prepare loss for model parameter
                loss = torch.stack(losses)
                # loss_batch = loss.dot(self.groupgroup_weights) - (self.reg / 2.) * self.groupgroup_weights.dot(torch.log(self.groupgroup_weights))
                loss_batch = loss.dot(self.groupgroup_weights)
                # backward the loss
                loss_batch.backward()
                # update model parameter
                self.optimizer.step()

                # update group_weights
                with torch.no_grad():
                    loss_ = loss.detach().clone()
                    loss_train += loss_
                    loss_freq += loss_
                    if freq_count % self.freq == 0:
                        loss_freq = 1. / max(1, freq_count) * loss_freq
                        new_groupgroup_weights = self.groupgroup_weights * torch.exp(
                            self.temp * self.scheduler.get_last_lr()[0] * loss_freq) * self.groupgroup_masks
                        self.groupgroup_weights = new_groupgroup_weights / new_groupgroup_weights.norm(p=1)
                        freq_count = 0
                        loss_freq = torch.zeros(train_writter.n_groups ** 2).to(self.device)

                self.scheduler.step()
                freq_count += 1

            # evaluate on the validation
            with torch.no_grad():
                self.model.eval()
                yy_val = self.model(X_val).view(-1)
                losses_val = []
                # loop through groups - no batch
                for (i, j) in product(range(val_writter.n_groups), range(val_writter.n_groups)):
                    # compute loss with lambda_beta
                    loss_group_val = self.criterion(yy_val[val_writter.grouplabel_ind[2*i]], yy_val[val_writter.grouplabel_ind[2*j+1]])
                    losses_val.append(loss_group_val)
                # prepare loss for model parameter
                loss_val = torch.stack(losses_val)
                loss_val_max = max(loss_val * self.groupgroup_masks)
                loss_val_weighted = loss_val.dot(self.groupgroup_weights)

                bestValLoss, bestepoch, temper = self.ckeck_n_save(loss_val_max, epoch, bestValLoss, bestepoch, temper,
                                                                   tol=self.tol, num_tempers=self.num_tempers)

                self.logger.log('Epoch:{} Train Loss:{:.4f} Val Loss:{:.4f} Val Max:{:.4f} Group Loss/Weight '.format(epoch, loss_batch, loss_val_weighted, loss_val_max),
                      newline=False, verbose=self.verbose)
                for (i, j) in product(range(val_writter.n_groups), range(val_writter.n_groups)):
                    self.logger.log('N-{} P-{}:{:.4f}/{:.2f} '.format(i, j, loss_val[
                        j + i * val_writter.n_groups].item(), self.groupgroup_weights[
                        j + i * val_writter.n_groups].item()), newline=False, verbose=self.verbose)
                self.logger.log("", verbose=self.verbose)

                # save the record
                self.bestmodel.eval()
                train_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_train)).squeeze(), min=1e-6, max=1-1e-6)
                val_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_val)).squeeze(), min=1e-6, max=1-1e-6)
                self.groupgroup_weights_rec.append(self.groupgroup_weights.detach().clone())
                train_recorder.update_stats(train_y_pred_proba, y_train)
                val_recorder.update_stats(val_y_pred_proba, y_val)

                if temper == 0:
                    self.logger.log('Early stopping at epoch {} and best epoch {}'.format(epoch, bestepoch), verbose=self.verbose)
                    break

        return train_recorder, val_recorder



MODEL_PATH_TO_CLASS = {"RWMIntraBCELearner": RWMBCELearner, "RWMInterBCELearner": RWMBCELearner, "RWMAllBCELearner": RWMBCELearner, 
                       "RWMIntraAUCLearner": RWMAUCLearner, "RWMInterAUCLearner": RWMAUCLearner, "RWMAllAUCLearner": RWMAUCLearner}

def load_rwm_model(model_name):
    """Returns an instantiation of the model."""
    return MODEL_PATH_TO_CLASS[model_name]

if __name__ == "__main__":

    from myutils.data_processor import load_db_by_name
    import numpy as np
    from myutils.np_writter import Writter

    db_name = "german"
    gp_name = "sex"
    model_name = "RWMIntraAUCLearner"

    data_train, data_test = load_db_by_name(db_name, gp_name)
    X_train, y_train, z_train = data_train
    y_train[y_train < 0.5] = 0.

    # n_groups = len(np.unique(z_tot)) # this causes potential bug that z_tot does not contain all the group
    X_test, y_test, z_test = data_test
    y_test[y_test < 0.5] = 0.

    # get number of groups
    n_groups = len(np.unique(np.concatenate((z_train, z_test))))

    # split train and validation
    # X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
    #     X_tot, y_tot, z_tot, test_size=args.validation_size, random_state=args.seed)

    n_samples, n_features = X_train.shape

    # Initialze stat and result writter
    train_writter = Writter(y_train, z_train)
    test_writter = Writter(y_test, z_test)

    Model = load_rwm_model(model_name)
    modelhat = Model(n_features, batch_size = 256, verbose=True).fit(X_train, y_train, z_train, X_test, y_test, z_test)