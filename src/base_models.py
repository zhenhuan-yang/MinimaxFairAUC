import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from myutils.torch_writter import TorchWritter
from myutils.model_helper import loss_helper, model_helper
from copy import deepcopy
from itertools import product
from myutils.model_helper import BaseClassifier

class BCELearner(BaseClassifier):

    def __init__(self, size, logger, beta=0.1, lr=.001, momentum=0., weight_decay=0.001, batch_size=16, temp=0.1, period=1,
                 multiplicative_factor=0.1, n_epochs=1000, random_state=42, tolerance=0.0001, num_tempers=3, 
                 verbose=False, device="cpu", save_path="", load_path=""):
        super(BCELearner, self).__init__()
        self.logger = logger
        self.device = device
        self.save_path = save_path
        self.load_path = load_path
        self.model = model_helper(size).to(self.device)
        self.bestmodel = model_helper(size).to(self.device)
        self.beta = beta
        self.lr = lr
        self.m = momentum
        self.wd = weight_decay
        self.bs = batch_size
        self.temp = temp
        self.p = period
        self.mf = multiplicative_factor
        self.n_epochs = n_epochs
        self.seed = random_state
        self.tol = tolerance
        self.num_tempers = num_tempers
        self.verbose = verbose

    def fit(self, train_data, train_writter, val_data, val_writter):

        # send everything to device
        X_train, y_train, group_neg_loaders, group_pos_loaders = self.np2torch(train_data, train_writter, get_loader=True)
        X_val, y_val, _, _ = self.np2torch(val_data, val_writter, get_loader=False)

        # initialize optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.m, weight_decay=self.wd)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.p, gamma=self.mf)
        # initialize loss
        self.criterion = loss_helper("bce")(k=None)

        train_recorder = TorchWritter(train_writter, beta=self.beta)
        val_recorder = TorchWritter(val_writter, beta=self.beta)

        # initialize stopping criterion
        bestValLoss = float("inf")
        temper = self.num_tempers
        bestepoch = 0

        self.model.train()  # Puts model in training mode so it updates itself!

        for epoch in range(1, 1 + self.n_epochs):

            loss_train = 0.
            self.model.train()

            for batch_idx, (group_neg_loader, group_pos_loader) in enumerate(zip(zip(*group_neg_loaders), zip(*group_pos_loaders))):

                X_batch_list = []
                n_batch_list = []
                
                # loop through groups
                for i in range(train_writter.n_groups):
                    # load the data from ith group
                    X_group_neg, = group_neg_loader[i]
                    X_group_pos, = group_pos_loader[i]
                    X_batch_list.append(X_group_neg)
                    X_batch_list.append(X_group_pos)
                    n_batch_list.append(X_group_neg.shape[0])
                    n_batch_list.append(X_group_pos.shape[0])
                
                X_batch = torch.cat(X_batch_list)
                n_batch_cs = torch.LongTensor(n_batch_list).cumsum(dim=0).tolist()

                self.optimizer.zero_grad()
                yy_pred = self.model(X_batch).view(-1)
                yy_batch = torch.zeros_like(yy_pred) # requires_grad=False
                for i in range(train_writter.n_groups):
                    yy_batch[n_batch_cs[2*i]:n_batch_cs[2*i+1]] = 1.
                loss_batch = self.criterion(yy_pred, yy_batch)
                loss_batch.backward()
                self.optimizer.step()

            loss_train += loss_batch.item()

            self.scheduler.step()

            with torch.no_grad():

                self.model.eval()
                y_pred_val = self.model.forward(X_val).view(-1)
                loss_val = self.criterion(y_pred_val, y_val)

                bestValLoss, bestepoch, temper = self.ckeck_n_save(loss_val, epoch, bestValLoss, bestepoch, temper,
                                                                   tol=self.tol, num_tempers=self.num_tempers)

                self.logger.log('Epoch: {} Train Loss: {:.6f} Val Loss: {:.6f}'.format(epoch, loss_batch, loss_val), verbose=self.verbose)

                if temper == 0:
                    self.logger.log('Early stopping at epoch {} and best epoch {}'.format(epoch, bestepoch), verbose=self.verbose)
                    break

        # save the record
        with torch.no_grad():
            self.bestmodel.eval()
            train_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_train)).squeeze(), min=1e-6, max=1-1e-6)
            val_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_val)).squeeze(), min=1e-6, max=1-1e-6)
            train_recorder.update_stats(train_y_pred_proba, y_train)
            val_recorder.update_stats(val_y_pred_proba, y_val)

        return train_recorder, val_recorder  # after trained

class AUCLearner(BaseClassifier):
    def __init__(self, size, logger, beta=0.1, lr=0.1, momentum=0., weight_decay=0.0001, batch_size=256, temp=0.1, period=1,
                 multiplicative_factor=0.1, n_epochs=1000, random_state=256, tolerance=1e-4, num_tempers=3, 
                 verbose=False, device="cpu", save_path="", load_path=""):
        super(AUCLearner, self).__init__()
        self.logger = logger
        self.device = device
        self.save_path = save_path
        self.load_path = load_path
        self.model = model_helper(size).to(self.device)
        if self.load_path:
            self.model.load_state_dict(torch.load(self.load_path))
        self.bestmodel = model_helper(size).to(self.device)
        self.beta = beta
        self.lr = lr
        self.m = momentum
        self.wd = weight_decay
        self.bs = batch_size
        self.temp = temp
        self.p = period
        self.mf = multiplicative_factor
        self.n_epochs = n_epochs
        self.seed = random_state
        self.tol = tolerance
        self.num_tempers = num_tempers
        self.verbose = verbose


    def fit(self, train_data, train_writter, val_data, val_writter):

        # initialize writter
        train_recorder = TorchWritter(train_writter, beta=self.beta)
        val_recorder = TorchWritter(val_writter, beta=self.beta)

        # send everything to device
        X_train, y_train, group_neg_loaders, group_pos_loaders = self.np2torch(train_data, train_writter, get_loader=True)
        X_val, y_val, _, _ = self.np2torch(val_data, val_writter, get_loader=False)
        
        bestValLoss = float("inf")
        temper = self.num_tempers
        bestepoch = 0

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.m, weight_decay=self.wd)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.p, gamma=self.mf)
        self.criterion = loss_helper('pauc')(k=None)

        for epoch in range(1, 1 + self.n_epochs):

            self.model.train()

            loss_train = 0.

            # multiple forward before backward
            # https://discuss.pytorch.org/t/multiple-model-forward-followed-by-one-loss-backward/20868
            for batch_idx, (group_neg_loader, group_pos_loader) in enumerate(zip(zip(*group_neg_loaders), zip(*group_pos_loaders))):

                X_batch_neg_list = []
                X_batch_pos_list = []
                n_batch_neg = 0
                n_batch_pos = 0
                # loop through groups
                for i in range(train_writter.n_groups):
                    # load the data from ith group
                    X_group_neg, = group_neg_loader[i]
                    X_group_pos, = group_pos_loader[i]
                    X_batch_neg_list.append(X_group_neg)
                    X_batch_pos_list.append(X_group_pos)
                    n_batch_neg += X_group_neg.shape[0]
                    n_batch_pos += X_group_pos.shape[0]
                
                X_batch_neg = torch.cat(X_batch_neg_list)
                X_batch_pos = torch.cat(X_batch_pos_list)
                X_batch = torch.cat((X_batch_neg, X_batch_pos))

                self.optimizer.zero_grad()
                # yy_neg_batch = self.model(X_neg_batch).view(-1)
                # yy_pos_batch = self.model(X_pos_batch).view(-1)
                yy_batch = self.model(X_batch).view(-1)
                yy_neg_batch, yy_pos_batch = torch.split(yy_batch, (n_batch_neg, n_batch_pos))
                loss_batch = self.criterion(yy_neg_batch, yy_pos_batch)
                loss_batch.backward()
                self.optimizer.step()  # get loss, use to update wts

                loss_train += loss_batch.item()

                self.scheduler.step()

            with torch.no_grad():
                self.model.eval()
                # yy_neg_val = self.model.forward(X_val[val_writter.labelindices[0]]).view(-1)
                # yy_pos_val = self.model.forward(X_val[val_writter.labelindices[1]]).view(-1)
                yy_val = self.model.forward(X_val).view(-1)
                yy_neg_val = yy_val[val_writter.label_ind[0]]
                yy_pos_val = yy_val[val_writter.label_ind[1]]
                loss_val = self.criterion(yy_neg_val, yy_pos_val)

                bestValLoss, bestepoch, temper = self.ckeck_n_save(loss_val, epoch, bestValLoss, bestepoch, temper, tol=self.tol, num_tempers=self.num_tempers)

                self.logger.log('Epoch: {} Train Loss: {:.6f} Val Loss: {:.6f}'.format(epoch, loss_batch, loss_val), verbose=self.verbose)

                if temper == 0:
                    self.logger.log('Early stopping at epoch {} and best epoch {}'.format(epoch, bestepoch), verbose=self.verbose)
                    break

        # save the record
        with torch.no_grad():
            self.bestmodel.eval()
            train_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_train)).squeeze(), min=1e-6,
                                             max=1 - 1e-6)
            val_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_val)).squeeze(), min=1e-6,
                                           max=1 - 1e-6)
            train_recorder.update_stats(train_y_pred_proba, y_train)
            val_recorder.update_stats(val_y_pred_proba, y_val)

        return train_recorder, val_recorder

class pAUCLearner(BaseClassifier):
    def __init__(self, size, logger, beta=0.1, lr=0.1, momentum=0., weight_decay=0.001, batch_size=1, temp=0.1, period=1,
                 multiplicative_factor=0.1, n_epochs=1000, random_state=42, tolerance=0.0001, num_tempers=3, 
                 verbose=False, device="cpu", save_path="", load_path=""):
        super(pAUCLearner, self).__init__()
        self.logger = logger
        self.device = device
        self.save_path = save_path
        self.load_path = load_path
        self.model = model_helper(size).to(self.device)
        if self.load_path:
            self.model.load_state_dict(torch.load(self.load_path))
        self.bestmodel = model_helper(size).to(self.device)
        self.beta = beta
        self.lr = lr
        self.m = momentum
        self.wd = weight_decay
        self.bs = batch_size
        self.temp = temp
        self.p = period
        self.mf = multiplicative_factor
        self.n_epochs = n_epochs
        self.seed = random_state
        self.tol = tolerance
        self.num_tempers = num_tempers
        self.verbose = verbose

    def fit(self, X_train, y_train, z_train, X_val, y_val, z_val):

        # initialize writter
        train_writter = TorchWritter(y_train, z_train)
        val_writter = TorchWritter(y_val, z_val)

        # send everything to device
        X_train, neg_train_loader, pos_train_loader, y_train, z_train, X_val, y_val, z_val = self.np2torch(X_train, y_train, z_train, X_val, y_val, z_val)

        # initialize validation criterion
        bestValLoss = float("inf")
        temper = self.num_tempers
        bestepoch = 0

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.m, weight_decay=self.wd)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.p, gamma=self.mf)
        self.criterion = loss_helper('pauc') # pAUCLoss class

        for epoch in range(1, 1 + self.n_epochs):

            self.model.train()

            loss_train = 0.

            # multiple forward before backward
            # https://discuss.pytorch.org/t/multiple-model-forward-followed-by-one-loss-backward/20868
            for batch_idx, (neg_batch, pos_batch) in enumerate(zip(neg_train_loader, pos_train_loader)):
                X_neg_batch, = neg_batch
                X_pos_batch, = pos_batch
                neg_bs = X_neg_batch.shape[0]
                pos_bs = X_pos_batch.shape[0]
                X_batch = torch.cat((X_neg_batch, X_pos_batch))
                self.optimizer.zero_grad()
                # yy_neg_batch = self.model(X_neg_batch).view(-1)
                # yy_pos_batch = self.model(X_pos_batch).view(-1)
                yy_batch = self.model(X_batch).view(-1)
                yy_neg_batch, yy_pos_batch = torch.split(yy_batch, (neg_bs, pos_bs))
                loss_batch = self.criterion(k=max(1, int(self.beta * neg_bs)))(yy_neg_batch, yy_pos_batch)
                loss_batch.backward()
                self.optimizer.step()
                # temp_loss = self.criterion(reduction='none', k=max(1, int(self.beta * neg_bs)))(yy_neg_batch, yy_pos_batch)
                # initial_loss = temp_loss.clone().detach()  # [n_pos, n_neg]
                # loss_batch = temp_loss.mean()
                # lambda_beta = initial_loss[:, -1].flatten()
                # lambda_beta = initial_loss.mean(dim=0)[-1]  # sorted=True

                loss_train += loss_batch.item()

                self.scheduler.step()

            with torch.no_grad():
                self.model.eval()
                yy_val = self.model.forward(X_val).view(-1)
                yy_neg_val = yy_val[val_writter.label_ind[0]]
                yy_pos_val = yy_val[val_writter.label_ind[1]]
                loss_val = self.criterion(k=max(1, int(self.beta * val_writter.n_neg)))(yy_neg_val, yy_pos_val)

                bestValLoss, bestepoch, temper = self.ckeck_n_save(loss_val, epoch, bestValLoss, bestepoch, temper,
                                                                   tol=self.tol, num_tempers=self.num_tempers)

                self.logger.log('Epoch: {} Train Loss: {:.6f} Val Loss: {:.6f}'.format(epoch, loss_batch, loss_val), verbose=self.verbose)

                if temper == 0:
                    self.logger.log('Early stopping at epoch {} and best epoch {}'.format(epoch, bestepoch), verbose=self.verbose)
                    break

        # save the record
        with torch.no_grad():
            self.bestmodel.eval()
            train_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_train)).squeeze(), min=1e-6,
                                             max=1 - 1e-6)
            val_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_val)).squeeze(), min=1e-6,
                                           max=1 - 1e-6)
            train_writter.update_stats(train_y_pred_proba, y_train, loss_train / (batch_idx + 1),
                                            beta=self.beta)
            val_writter.update_stats(val_y_pred_proba, y_val, loss_val.item(), beta=self.beta)

        return self

class ThrespAUCLearner(BaseClassifier):

    def __init__(self, size, logger, beta=0.1, lr=0.1, momentum=0., weight_decay=0.001, batch_size=1, temp=0.1, period=1,
                 multiplicative_factor=0.1, n_epochs=1000, random_state=42, tolerance=1e-4, num_tempers=3, 
                 verbose=False, device="cpu", save_path="", load_path=""):
        super(ThrespAUCLearner, self).__init__()
        self.logger = logger
        self.device = device
        self.save_path = save_path
        self.load_path = load_path
        self.model = model_helper(size).to(self.device)
        if self.load_path:
            self.model.load_state_dict(torch.load(self.load_path))
        self.bestmodel = model_helper(size).to(self.device)
        self.beta = beta
        self.lr = lr
        self.m = momentum
        self.wd = weight_decay
        self.p = period
        self.mf = multiplicative_factor
        self.bs = batch_size
        self.temp = temp
        self.n_epochs = n_epochs
        self.seed = random_state
        self.tol = tolerance
        self.num_tempers = num_tempers
        self.verbose = verbose


    def fit(self, X_train, y_train, z_train, X_val, y_val, z_val):

        # initialize writter
        train_writter = TorchWritter(y_train, z_train)
        val_writter = TorchWritter(y_val, z_val)

        # send everything to device
        X_train = torch.from_numpy(X_train).to(self.device)
        y_train = torch.from_numpy(y_train).to(self.device)
        z_train = torch.from_numpy(z_train).to(self.device)
        X_val = torch.from_numpy(X_val).to(self.device)
        y_val = torch.from_numpy(y_val).to(self.device)
        z_val = torch.from_numpy(z_val).to(self.device)


        neg_train_set = TensorDataset(X_train[train_writter.labelindices[0]])
        X_pos = X_train[train_writter.labelindices[1]]
        pos_train_set = TensorDataset(torch.arange(train_writter.n_pos).to(self.device))


        # sample data with
        neg_train_loader = DataLoader(neg_train_set, batch_size=max(1, int(self.bs * train_writter.n_neg / train_writter.n_samples)), drop_last=True)
        pos_train_loader = DataLoader(pos_train_set, batch_size=max(1, int(self.bs * train_writter.n_pos / train_writter.n_samples)), drop_last=True)

        # use topk loss first
        self.criterion = loss_helper('pauc')(reduction='none', k=max(1, int(self.beta * self.bs * train_writter.n_neg / train_writter.n_samples)))

        bestValLoss = float("inf")
        temper = self.num_tempers
        bestepoch = 0

        # initialize for each positive sample
        lambda_beta = torch.zeros(train_writter.n_pos, requires_grad=True, device=self.device)
        lambda_agg_grad = torch.zeros(train_writter.n_pos, device=self.device)


        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.m, weight_decay=self.wd)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.p, gamma=self.mf)

        # # initialization for better lambda
        init_epochs = 3

        for epoch in range(1, init_epochs + 1):

            self.model.train()

            for batch_ids, (neg_batch, pos_batch) in enumerate(zip(neg_train_loader, pos_train_loader)):
                X_neg_batch, = neg_batch
                pos_batch_idx, = pos_batch
                X_pos_batch = X_pos[pos_batch_idx]
                neg_bs = X_neg_batch.shape[0]
                pos_bs = X_pos_batch.shape[0]

                X_batch = torch.cat((X_neg_batch, X_pos_batch))
                self.optimizer.zero_grad()
                # yy_neg_batch = self.model(X_neg_batch).view(-1)
                # yy_pos_batch = self.model(X_pos_batch).view(-1)
                yy_batch = self.model(X_batch).view(-1)
                yy_neg_batch, yy_pos_batch = torch.split(yy_batch, (neg_bs, pos_bs))
                temp_loss = self.criterion(yy_neg_batch, yy_pos_batch)
                loss = temp_loss.mean()
                initial_loss = temp_loss.clone().detach() # [pos_bs, neg_bs] tensor

                loss.backward()
                self.optimizer.step()  # get loss, use to update wts
                # return the kth largest negative
                # only on the selected index
                lambda_beta.data[pos_batch_idx] = initial_loss[:, -1].flatten()

        # switch back to threshold loss
        self.criterion = loss_helper('threspauc')
        self.criterion_val = loss_helper('pauc')(k=max(1, int(self.beta * val_writter.n_neg)))

        for epoch in range(1, 1 + self.n_epochs):

            self.model.train()

            loss_train = 0.

            for batch_idx, (neg_batch, pos_batch) in enumerate(zip(neg_train_loader, pos_train_loader)):
                X_neg_batch, = neg_batch
                pos_batch_idx, = pos_batch
                X_pos_batch = X_pos[pos_batch_idx]
                neg_bs = X_neg_batch.shape[0]
                pos_bs = X_pos_batch.shape[0]
                X_batch = torch.cat((X_neg_batch, X_pos_batch))
                self.optimizer.zero_grad()
                # yy_neg_batch = self.model(X_neg_batch).view(-1)
                # yy_pos_batch = self.model(X_pos_batch).view(-1)
                yy_batch = self.model(X_batch).view(-1)
                yy_neg_batch, yy_pos_batch = torch.split(yy_batch, (neg_bs, pos_bs))

                loss_batch = self.criterion(k=max(1, int(self.beta * neg_bs)))(yy_neg_batch, yy_pos_batch, thres=lambda_beta[pos_batch_idx])
                lambda_beta.retain_grad()  # retain the grad for non-leaf variables
                loss_batch.backward()
                self.optimizer.step()  # get loss, use to update wts

                lambda_beta.grad.data += self.wd * lambda_beta.data
                lambda_agg_grad = self.m * lambda_agg_grad + lambda_beta.grad.data
                lambda_beta.data = lambda_beta.data - self.scheduler.get_last_lr()[0] * lambda_agg_grad
                lambda_beta.grad.data.zero_()  # clear the grad

                loss_train += loss_batch.item()

                self.scheduler.step()

            with torch.no_grad():
                self.model.eval()
                yy_val = self.model.forward(X_val).view(-1)
                yy_neg_val = yy_val[val_writter.labelindices[0]]
                yy_pos_val = yy_val[val_writter.labelindices[1]]
                loss_val = self.criterion_val(yy_neg_val, yy_pos_val)

                bestValLoss, bestepoch, temper = self.ckeck_n_save(loss_val, epoch, bestValLoss, bestepoch, temper,
                                                                   tol=self.tol, num_tempers=self.num_tempers)

                self.logger.log('Epoch: {} Train Loss: {:.6f} Val Loss: {:.6f}'.format(epoch, loss_batch.item(), loss_val), verbose=self.verbose)

                if temper == 0:
                    self.logger.log('Early stopping at epoch {} and best epoch {}'.format(epoch, bestepoch), verbose=self.verbose)
                    break

        # save the record
        with torch.no_grad():
            self.bestmodel.eval()
            train_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_train)).squeeze(), min=1e-6,
                                             max=1 - 1e-6)
            val_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_val)).squeeze(), min=1e-6,
                                           max=1 - 1e-6)
            train_writter.update_stats(train_y_pred_proba, y_train, loss_train / (batch_idx + 1),
                                            beta=self.beta)
            val_writter.update_stats(val_y_pred_proba, y_val, loss_val.item(), beta=self.beta)

        # save the optimal lambda
        # self.lambda_beta = lambda_beta.data.numpy()

        return self

class SingleThrespAUCLearner(BaseClassifier):

    def __init__(self, size, logger, beta=0.1, lr=0.1, momentum=0., weight_decay=0.001, batch_size=1, temp=0.1, period=1,
                 multiplicative_factor=0.1, n_epochs=1000, random_state=42, tolerance=1e-4, num_tempers=3, 
                 verbose=False, device="cpu", save_path="", load_path=""):
        super(SingleThrespAUCLearner, self).__init__()
        self.logger = logger
        self.device = device
        self.save_path = save_path
        self.load_path = load_path
        self.model = model_helper(size).to(self.device)
        if self.load_path:
            self.model.load_state_dict(torch.load(self.load_path))
        self.bestmodel = model_helper(size).to(self.device)
        self.beta = beta
        self.lr = lr
        self.m = momentum
        self.wd = weight_decay
        self.p = period
        self.mf = multiplicative_factor
        self.bs = batch_size
        self.temp = temp
        self.n_epochs = n_epochs
        self.seed = random_state
        self.tol = tolerance
        self.num_tempers = num_tempers
        self.verbose = verbose


    def fit(self, X_train, y_train, z_train, X_val, y_val, z_val):

        # initialize writter
        train_writter = TorchWritter(y_train, z_train)
        val_writter = TorchWritter(y_val, z_val)

        # send everything to device
        X_train, neg_train_loader, pos_train_loader, y_train, z_train, X_val, y_val, z_val = self.np2torch(X_train, y_train, z_train, X_val, y_val, z_val)
        
        # use topk loss first
        self.criterion = loss_helper('pauc')(reduction='none', k=max(1, int(self.beta * self.bs * train_writter.n_neg / train_writter.n_samples)))

        bestValLoss = float("inf")
        temper = self.num_tempers
        bestepoch = 0

        # initialize for each positive sample
        lambda_beta = torch.zeros(1, requires_grad=True).to(self.device)
        lambda_agg_grad = torch.zeros(1).to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.m, weight_decay=self.wd)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.p, gamma=self.mf)
        # initialization for better lambda
        init_epochs = 3

        for epoch in range(1, init_epochs + 1):

            self.model.train()

            for batch_ids, (neg_batch, pos_batch) in enumerate(zip(neg_train_loader, pos_train_loader)):
                X_neg_batch, = neg_batch
                X_pos_batch, = pos_batch
                neg_bs = X_neg_batch.shape[0]
                pos_bs = X_pos_batch.shape[0]
                X_batch = torch.cat((X_neg_batch, X_pos_batch))
                self.optimizer.zero_grad()
                # yy_neg_batch = self.model(X_neg_batch).view(-1)
                # yy_pos_batch = self.model(X_pos_batch).view(-1)
                yy_batch = self.model(X_batch).view(-1)
                yy_neg_batch, yy_pos_batch = torch.split(yy_batch, (neg_bs, pos_bs))
                temp_loss = self.criterion(yy_neg_batch, yy_pos_batch)
                loss = temp_loss.mean()
                initial_loss = temp_loss.clone().detach()
                loss.backward()
                self.optimizer.step()  # get loss, use to update wts
        lambda_beta.data = initial_loss.mean(dim=0)[-1].flatten() # dim/axis collapse the specified dim/axis

        # switch back to threshold loss
        self.criterion = loss_helper('singlethrespauc')

        for epoch in range(1, 1 + self.n_epochs):

            self.model.train()

            loss_train = 0.

            for batch_idx, (neg_batch, pos_batch) in enumerate(zip(neg_train_loader, pos_train_loader)):
                X_neg_batch, = neg_batch
                X_pos_batch, = pos_batch
                neg_bs = X_neg_batch.shape[0]
                pos_bs = X_pos_batch.shape[0]
                X_batch = torch.cat((X_neg_batch, X_pos_batch))
                self.optimizer.zero_grad()
                # yy_neg_batch = self.model(X_neg_batch).view(-1)
                # yy_pos_batch = self.model(X_pos_batch).view(-1)
                yy_batch = self.model(X_batch).view(-1)
                yy_neg_batch, yy_pos_batch = torch.split(yy_batch, (neg_bs, pos_bs))
                loss_batch = self.criterion(k=max(1, int(self.beta * neg_bs)))(yy_neg_batch, yy_pos_batch, thres=lambda_beta)
                lambda_beta.retain_grad()  # retain the grad for non-leaf variables
                loss_batch.backward()
                self.optimizer.step()  # get loss, use to update wts

                lambda_beta.grad.data += self.wd * lambda_beta.data
                lambda_agg_grad = self.m * lambda_agg_grad + lambda_beta.grad.data
                lambda_beta.data = lambda_beta.data - self.scheduler.get_last_lr()[0] * lambda_agg_grad
                lambda_beta.grad.data.zero_()  # clear the grad

                loss_train += loss_batch.item()

                self.scheduler.step()

            with torch.no_grad():
                self.model.eval()
                yy_val = self.model.forward(X_val).view(-1)
                yy_neg_val = yy_val[val_writter.labelindices[0]]
                yy_pos_val = yy_val[val_writter.labelindices[1]]
                loss_val = self.criterion(k=max(1, int(self.beta *val_writter.n_neg)))(yy_neg_val, yy_pos_val, thres=lambda_beta).squeeze()

                bestValLoss, bestepoch, temper = self.ckeck_n_save(loss_val, epoch, bestValLoss, bestepoch, temper,
                                                                   tol=self.tol, num_tempers=self.num_tempers)

                self.logger.log('Epoch: {} Train Loss: {:.6f} Val Loss: {:.6f}'.format(epoch, loss_batch.item(), loss_val), verbose=self.verbose)

                if temper == 0:
                    self.logger.log('Early stopping at epoch {} and best epoch {}'.format(epoch, bestepoch), verbose=self.verbose)
                    break

        # save the record
        with torch.no_grad():
            self.bestmodel.eval()
            train_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_train)).squeeze(), min=1e-6,
                                             max=1 - 1e-6)
            val_y_pred_proba = torch.clamp(torch.sigmoid(self.bestmodel(X_val)).squeeze(), min=1e-6,
                                           max=1 - 1e-6)
            train_writter.update_stats(train_y_pred_proba, y_train, loss_train / (batch_idx + 1),
                                            beta=self.beta)
            val_writter.update_stats(val_y_pred_proba, y_val, loss_val.item(), beta=self.beta)

        # save the optimal lambda
        # self.lambda_beta = lambda_beta.data.numpy()

        return self

MODEL_PATH_TO_CLASS = {"LogisticRegression": LogisticRegression, "BCELearner": BCELearner, "ThrespAUCLearner": ThrespAUCLearner,
                       "pAUCLearner": pAUCLearner, "AUCLearner": AUCLearner, "SingleThrespAUCLearner": SingleThrespAUCLearner}

def load_model(model_name):
    """Returns an instantiation of the model."""
    return MODEL_PATH_TO_CLASS[model_name]

if __name__ == "__main__":

    from myutils.data_processor import load_db_by_name
    import numpy as np
    from myutils.np_writter import Writter

    db_name = "adult"
    gp_name = "sex"
    model_name = "SingleThrespAUCLearner"

    train_data, val_data = load_db_by_name(db_name, gp_name)
    X_tr, y_tr, z_tr = train_data
    y_tr[y_tr < 0.5] = 0.

    # n_groups = len(np.unique(z_tot)) # this causes potential bug that z_tot does not contain all the group
    X_te, y_te, z_te = val_data
    y_te[y_te < 0.5] = 0.

    # get number of groups
    n_groups = len(np.unique(np.concatenate((z_tr, z_te))))

    # split train and validation
    # X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
    #     X_tot, y_tot, z_tot, test_size=args.validation_size, random_state=args.seed)

    n_samples, n_features = X_tr.shape
    test_n_samples, test_n_features = X_te.shape
    assert n_features == test_n_features, "Inconsistent train and test features!"

    # Initialze stat and result writter
    train_writter = Writter(y_tr, z_tr)
    val_writter = Writter(y_te, z_te)

    Model = load_model(model_name)
    modelhat = Model(n_features, verbose=True).fit(X_tr, y_tr, z_tr, X_te, y_te, z_te)