import torch
import numpy as np
from itertools import product
from myutils.roc_optimizer import mis, bce, pauc, lauc

class TorchWritter:
    '''
    For writting stat and result by experiments
    '''
    def __init__(self, writter, beta=0.1, n_tempers=5):
        '''
        Initialize indexes in tensor for use later
        This way is faster when cuda is enabled
        '''
        self.n_groups = writter.n_groups
        self.label_ind = writter.label_ind
        self.group_ind = writter.group_ind
        self.grouplabel_ind = writter.grouplabel_ind
        self.n_neg = writter.n_neg
        self.beta = beta
        self.n_tempers = n_tempers
        # initialize result
        self.mis_errors = []
        self.bce_errors = []
        self.auc_errors = []
        self.pauc_errors = []
        self.lauc_errors = []
        self.mis_grouperrs = []
        self.bce_grouperrs = []
        self.auc_grouperrs = []
        self.pauc_grouperrs = []
        self.lauc_grouperrs = []
        for i in range(self.n_groups):
            self.mis_grouperrs.append([])
            self.bce_grouperrs.append([])
        for (i, j) in product(range(self.n_groups), range(self.n_groups)):
            self.auc_grouperrs.append([])
            self.pauc_grouperrs.append([])
            self.lauc_grouperrs.append([])
        self.y_pred_proba = []
        # baseline record the overall loss
        # fairness record individual group loss
        self.loss = []

    def update_stats(self, y_pred_proba, y):
        with torch.no_grad():
            self.y_pred_proba.append(y_pred_proba)
            # self.loss.append(loss)
            # Compute the overall errors by t
            self.mis_errors.append(mis(y_pred_proba, y))
            self.bce_errors.append(bce(y_pred_proba, y))
            self.auc_errors.append(pauc(y_pred_proba[self.label_ind[0]], y_pred_proba[self.label_ind[1]], k=None))
            self.pauc_errors.append(pauc(y_pred_proba[self.label_ind[0]], y_pred_proba[self.label_ind[1]], k=max(1, int(self.beta * self.n_neg)), norm=True))
            self.lauc_errors.append(lauc(y_pred_proba[self.label_ind[0]], y_pred_proba[self.label_ind[1]]))
            for i in range(self.n_groups):
                self.mis_grouperrs[i].append(mis(y_pred_proba[self.group_ind[i]], y[self.group_ind[i]]))
                self.bce_grouperrs[i].append(bce(y_pred_proba[self.group_ind[i]], y[self.group_ind[i]]))
            for (i, j) in product(range(self.n_groups), range(self.n_groups)):
                self.auc_grouperrs[j+i*self.n_groups].append(
                    pauc(y_pred_proba[self.grouplabel_ind[2*i]], y_pred_proba[self.grouplabel_ind[2*j+1]], k=None))
                self.pauc_grouperrs[j+i*self.n_groups].append(
                    pauc(y_pred_proba[self.grouplabel_ind[2*i]], y_pred_proba[self.grouplabel_ind[2*j+1]], k=max(1, int(self.beta * len(self.grouplabel_ind[2*i]))), norm=True))
                self.lauc_grouperrs[j+i*self.n_groups].append(
                    lauc(y_pred_proba[self.grouplabel_ind[2*i]], y_pred_proba[self.grouplabel_ind[2*j+1]]))


    