import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from abc import ABC
import scipy
import numpy as np
import torch.optim as optim

class Linear(nn.Module):
    def __init__(self, n_feat):
        super(Linear, self).__init__()
        self.n_feat = n_feat
        self.fc = nn.Linear(self.n_feat, 1, bias=True)
        # nn.init.xavier_normal_(self.fc.weight)
    def forward(self, x):
        out = self.fc(x)
        return out

class MLP(nn.Module):
    """
    A feedforward NN in pytorch using ReLU activiation functions between all layers
    The output layer uses BatchNorm
    Supports an arbitrary number of hidden layers
    """

    def __init__(self, h_sizes):
        """
        :param h_sizes: input sizes for each hidden layer (including the first)
        :param out_size: defaults to 1 for binary and represents the (positive class probability?)
        :param task: 'classification' or 'regression'
        """
        super(MLP, self).__init__()

        # Hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1], bias=True))

        # Output layer
        self.out = nn.Linear(h_sizes[-1], 1, bias=True)
        self.relu = torch.nn.ReLU()
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        # Feedforward
        for layer in self.hidden:
            x = self.relu(layer(x))
        output = self.bn(self.out(x))
        return output

class BaseClassifier:
    """
    Wrapper class so torch module looks like an sklearn model
    """

    def __init__(self):
        pass

    def np2torch(self, X_train, train_writter, get_loader=True):

        X_train_list = []

        group_neg_loaders = []
        group_pos_loaders = []

        for i in range(train_writter.n_groups):

            try: 
                X_train_neg = torch.from_numpy(X_train[2*i]).to(self.device)
                X_train_pos = torch.from_numpy(X_train[2*i+1]).to(self.device)

            except TypeError:

                # values = X_train[2*i].data
                # indices = np.vstack((X_train[2*i].row, X_train[2*i].col))
                # ind = torch.LongTensor(indices)
                # v = torch.FloatTensor(values)
                # shape = X_train[2*i].shape
                # X_train_neg = torch.sparse.FloatTensor(ind, v, torch.Size(shape)).to(self.device)
                
                # values = X_train[2*i+1].data
                # indices = np.vstack((X_train[2*i+1].row, X_train[2*i+1].col))
                # ind = torch.LongTensor(indices)
                # v = torch.FloatTensor(values)
                # shape = X_train[2*i+1].shape
                # X_train_pos = torch.sparse.FloatTensor(ind, v, torch.Size(shape)).to(self.device)

                X_train_neg = torch.FloatTensor(scipy.sparse.coo_matrix.todense(X_train[2*i])).to(self.device)
                X_train_pos = torch.FloatTensor(scipy.sparse.coo_matrix.todense(X_train[2*i+1])).to(self.device)

            X_train_list.append(X_train_neg)
            X_train_list.append(X_train_pos)

            if get_loader:

                group_neg_train_set = TensorDataset(X_train_neg)
                group_pos_train_set = TensorDataset(X_train_pos)

                group_neg_loaders.append(DataLoader(group_neg_train_set, batch_size=max(1, int(self.bs * train_writter.n_grouplabel[2*i] / train_writter.n_samples)), drop_last=True))
                group_pos_loaders.append(DataLoader(group_pos_train_set, batch_size=max(1, int(self.bs * train_writter.n_grouplabel[2*i+1] / train_writter.n_samples)), drop_last=True))
            
        X_train = torch.cat(X_train_list)
        with torch.no_grad():

            self.model.eval()
            y_pred_train = self.model.forward(X_train).view(-1)
            y_train = torch.zeros_like(y_pred_train)
            y_train[train_writter.label_ind[1]] = 1.


        return X_train, y_train, group_neg_loaders, group_pos_loaders

    def ckeck_n_save(self, loss, epoch, bestValLoss, bestepoch, temper, tol=0.0001, num_tempers=3):

        # https://stackoverflow.com/questions/67161171/how-to-store-model-state-dict-in-a-temp-var-for-later-use
        # https://discuss.pytorch.org/t/deepcopy-vs-load-state-dict/61527

        if epoch == 1 or loss.item() < bestValLoss - tol:
            bestepoch = epoch
            temper = num_tempers
            bestValLoss = loss.item()
            self.bestmodel.load_state_dict(self.model.state_dict())
            # self.bestmodel = deepcopy(self.model.state_dict())
            if self.save_path:
                torch.save(self.bestmodel.state_dict(), self.save_path)
        else:
            temper -= 1

        return bestValLoss, bestepoch, temper

    def predict_proba(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: Column vector of prediction probabilities, one for each row (instance) in X
        """
        self.bestmodel.eval()
        with torch.no_grad():
            return torch.sigmoid(self.bestmodel(torch.from_numpy(X).to(self.device))).cpu().numpy().squeeze() # Apply sigmoid manually

    def predict(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: Binary predictions for each instance of X
        """
        return self.predict_proba(X) > 0.5 # Converts probabilistic predictions into binary ones

    def score(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: score, one for each row (instance) in X
        """
        self.bestmodel.eval()
        with torch.no_grad():
            return self.bestmodel(torch.from_numpy(X).to(self.device)).cpu().numpy().squeeze()

class AbsClassifier:
    """
    Wrapper class so torch module looks like an sklearn model
    """

    def __init__(self):
        pass

    def stepsize_helper(self):
        if self.rtype == 'const':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=self.p, gamma=self.mf)
        elif self.rtype == 'sqrt':
            lambda_lr = lambda epoch: 1.0 / ((1.0 + epoch) ** 0.5)
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        else: 
            lambda_lr = lambda epoch: 1.0 / (1.0 + epoch) 
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)


    def np2torch(self, X_train, train_writter, get_loader=True):

        X_train_list = []

        group_neg_loaders = []
        group_pos_loaders = []

        for i in range(train_writter.n_groups):

            try: 
                X_train_neg = torch.from_numpy(X_train[2*i]).to(self.device)
                X_train_pos = torch.from_numpy(X_train[2*i+1]).to(self.device)

            except TypeError:

                values = X_train[2*i].data
                indices = np.vstack((X_train[2*i].row, X_train[2*i].col))
                ind = torch.LongTensor(indices)
                v = torch.FloatTensor(values)
                shape = X_train[2*i].shape
                X_train_neg = torch.sparse.FloatTensor(ind, v, torch.Size(shape)).to(self.device)
                
                values = X_train[2*i+1].data
                indices = np.vstack((X_train[2*i+1].row, X_train[2*i+1].col))
                ind = torch.LongTensor(indices)
                v = torch.FloatTensor(values)
                shape = X_train[2*i+1].shape
                X_train_pos = torch.sparse.FloatTensor(ind, v, torch.Size(shape)).to(self.device)

            X_train_list.append(X_train_neg)
            X_train_list.append(X_train_pos)

            if get_loader:

                group_neg_train_set = TensorDataset(X_train_neg)
                group_pos_train_set = TensorDataset(X_train_pos)

                group_neg_loaders.append(DataLoader(group_neg_train_set, batch_size=max(1, int(self.bs * train_writter.n_grouplabel[2*i] / train_writter.n_samples)), drop_last=True))
                group_pos_loaders.append(DataLoader(group_pos_train_set, batch_size=max(1, int(self.bs * train_writter.n_grouplabel[2*i+1] / train_writter.n_samples)), drop_last=True))
        
        X_train = torch.cat(X_train_list)
        # get label because of bce
        with torch.no_grad():

            self.model.eval()
            y_pred_train = self.model.forward(X_train).view(-1)
            y_train = torch.zeros_like(y_pred_train)
            y_train[train_writter.label_ind[1]] = 1.

        return X_train, y_train, group_neg_loaders, group_pos_loaders

    def ckeck_n_save(self, loss, epoch, bestValLoss, bestepoch, temper, tol=0.0001, num_tempers=3):

        # https://stackoverflow.com/questions/67161171/how-to-store-model-state-dict-in-a-temp-var-for-later-use
        # https://discuss.pytorch.org/t/deepcopy-vs-load-state-dict/61527
        if epoch == 1 or abs(loss - bestValLoss) > tol:
            bestepoch = epoch
            temper = num_tempers
            bestValLoss = loss
            self.bestmodel.load_state_dict(self.model.state_dict())
            # self.bestmodel = deepcopy(self.model.state_dict())
            if self.save_path:
                torch.save(self.bestmodel.state_dict(), self.save_path)
        else:
            temper -= 1

        return bestValLoss, bestepoch, temper

    def predict_proba(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: Column vector of prediction probabilities, one for each row (instance) in X
        """
        with torch.no_grad():
            self.bestmodel.eval()
            return torch.sigmoid(self.bestmodel(torch.from_numpy(X)).to(self.device)).cpu().numpy().squeeze() # Apply sigmoid manually

    def predict(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: Binary predictions for each instance of X
        """
        return self.predict_proba(X) > 0.5  # Converts probabilistic predictions into binary ones

    def score(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: score, one for each row (instance) in X
        """
        self.bestmodel.eval()
        with torch.no_grad():
            return self.bestmodel(torch.from_numpy(X).to(self.device)).cpu().numpy().squeeze()

class AuxBCEWithLogitLoss(nn.Module):

    def __init__(self, reduction='mean', weight=None, k=None):
        super(AuxBCEWithLogitLoss, self).__init__()
        self.r = reduction
        self.w = weight
        self.k = k

    def forward(self, y_pred, y):
        y_stab = torch.clamp(torch.sigmoid(y_pred), min=1e-6, max=1-1e-6)
        if self.w is not None:
            bce = - self.w * (y * torch.log(y_stab) + (1. - y) * torch.log(1. - y_stab))
        else:
            bce = - (y * torch.log(y_stab) + (1. - y) * torch.log(1. - y_stab))
        if self.k:
            bce = torch.topk(bce, self.k)[0]
        if self.r == 'none':
            return bce
        elif self.r == 'mean':
            return bce.mean()
        elif self.r == 'sum':
            return bce.sum()
        else:
            return

class ThresTopkLoss(nn.Module):

    def __init__(self, weight=None, k=1):
        super(ThresTopkLoss, self).__init__()
        self.sw = weight
        self.k = k

    def forward(self, y_pred, y, thres=0.):
        y_stab = torch.clamp(torch.sigmoid(y_pred.view(-1)), min=1e-6, max=1-1e-6)
        if self.sw is not None:
            bce = - self.sw * (y.view(-1) * torch.log(y_stab) + (1. - y.view(-1)) * torch.log(1. - y_stab))
        else:
            bce = - (y.view(-1) * torch.log(y_stab) + (1. - y.view(-1)) * torch.log(1. - y_stab))

        # thres = torch.topk(bce, self.k)[0][-1] # recovers direct topk result
        loss = torch.clamp(bce - thres, min=0.).sum() / self.k + thres

        return loss

class ThrespAUCLoss(nn.Module):

    def __init__(self, k=1, norm=True):
        super(ThrespAUCLoss, self).__init__()
        self.k = k
        self.norm = norm

    def forward(self, score_neg, score_pos, thres=0.):

        y_pos = torch.reshape(score_pos, (-1, 1))
        y_neg = torch.reshape(score_neg, (-1, 1))
        yy = y_pos - torch.transpose(y_neg, 0, 1)
        bce = - torch.log(torch.clamp(torch.sigmoid(yy), min=1e-6, max=1-1e-6))
        # thres = torch.reshape(thres, (-1, 1))
        if self.norm:
            loss = (torch.clamp(bce - thres.reshape(-1, 1), min=0.).sum() / self.k + thres.sum()) / len(score_pos)
        else:
            loss = (torch.clamp(bce - thres.reshape(-1, 1), min=0.).sum() / len(score_neg) + thres.sum()) / len(score_pos)
        return loss

class SingleThrespAUCLoss(nn.Module):

    def __init__(self, k=1, norm=True):
        super(SingleThrespAUCLoss, self).__init__()
        self.k = k
        self.norm = norm

    def forward(self, score_neg, score_pos, thres=0.):

        y_pos = torch.reshape(score_pos, (-1, 1))
        y_neg = torch.reshape(score_neg, (-1, 1))
        yy = y_pos - torch.transpose(y_neg, 0, 1)
        bce = - torch.log(torch.clamp(torch.sigmoid(yy), min=1e-6, max=1-1e-6))
        if self.norm:
            loss = torch.clamp(bce.mean(dim=0) - thres, min=0.).sum() / self.k + thres
        else:
            loss = torch.clamp(bce.mean(dim=0) - thres, min=0.).sum() / len(score_neg) + thres
        return loss

class SingleThresxpAUCLoss(nn.Module):

    def __init__(self, k=1):
        super(SingleThresxpAUCLoss, self).__init__()
        self.k = k

    def forward(self, score_neg, score_pos, thres=0.):

        y_pos = torch.reshape(score_pos, (-1, 1))
        y_neg = torch.reshape(score_neg, (-1, 1))
        yy = y_pos - torch.transpose(y_neg, 0, 1)
        bce = - torch.log(torch.clamp(torch.sigmoid(yy), min=1e-6, max=1-1e-6))
        loss = torch.clamp(bce.mean(dim=0) - thres, min=0.).sum() / self.k + thres

        return loss

class pAUCLoss(nn.Module):

    def __init__(self, k=1, reduction='mean', norm=False):
        super(pAUCLoss, self).__init__()
        self.reduction = reduction
        self.norm = norm
        self.k = k

    def forward(self, score_neg, score_pos):

        y_pos = torch.reshape(score_pos, (-1, 1))
        y_neg = torch.reshape(score_neg, (-1, 1))
        yy = y_pos - torch.transpose(y_neg, 0, 1) # [n_pos, n_neg] 2d-tensor
        bce = - torch.log(torch.clamp(torch.sigmoid(yy), min=1e-6, max=1-1e-6)) # [n_pos, n_neg] 2d-tensor
        if self.k:
            bce = torch.topk(bce, self.k)[0]
        if self.reduction == 'mean':
            if self.norm:
                return bce.mean()
            else:
                return bce.sum() / (len(score_pos) * len(score_neg))
        elif self.reduction == 'sum':
            return bce.sum()
        elif self.reduction == 'none':
            return bce
        else:
            return

class xAUCLoss(nn.Module):

    def __init__(self, k=1, reduction='mean'):
        super(xAUCLoss, self).__init__()
        self.reduction = reduction
        self.k = k

    def forward(self, score_neg, score_pos):

        y_pos = torch.reshape(score_pos, (-1, 1))
        y_neg = torch.reshape(score_neg, (-1, 1))
        yy = y_pos - torch.transpose(y_neg, 0, 1) # [n_pos, n_neg] matrix
        bce = - torch.log(torch.clamp(torch.sigmoid(yy), min=1e-6, max=1-1e-6))
        if self.k:
            bce = torch.topk(bce, self.k)[0]
        if self.reduction == 'mean':
            return bce.mean()
        elif self.reduction == 'sum':
            return bce.sum()
        elif self.reduction == 'none':
            return bce
        else:
            return

def model_helper(size):
    if isinstance(size, int):
        return Linear(size)
    elif isinstance(size, list):
        return MLP(size)
    else:
        raise ValueError('Invalid size type!')

def loss_helper(loss_type):
    if loss_type == 'bce':
        return AuxBCEWithLogitLoss
    elif loss_type == 'topk':
        return ThresTopkLoss
    elif loss_type == 'xauc':
        return xAUCLoss
    elif loss_type == 'pauc':
        return pAUCLoss
    elif loss_type == 'threspauc':
        return ThrespAUCLoss
    elif loss_type == 'singlethrespauc':
        return SingleThrespAUCLoss
    elif loss_type == 'singlethresxpauc':
        return SingleThresxpAUCLoss
    else:
        raise ValueError('Invalid loss type!')
