import numpy as np
import torch

class Writter:
    '''
    For writting stat and result by experiments
    '''
    def __init__(self):
        pass

    def torch2np(self, torch_writter):
        
        if len(torch_writter.mis_errors) > 1:

            # self.n_steps = len(torch_writter.loss) - torch_writter.n_tempers
            self.n_steps = len(torch_writter.mis_errors) - torch_writter.n_tempers
            self.y_pred_proba = torch.stack(torch_writter.y_pred_proba).cpu().numpy()[:self.n_steps, :]
            # self.loss = torch.stack(torch_writter.loss).cpu().numpy()[:self.n_steps]
            self.mis_errors = np.array(torch_writter.mis_errors)[:self.n_steps]
            self.bce_errors = np.array(torch_writter.bce_errors)[:self.n_steps]
            self.auc_errors = np.array(torch_writter.auc_errors)[:self.n_steps]
            self.pauc_errors = np.array(torch_writter.pauc_errors)[:self.n_steps]
            self.lauc_errors = np.array(torch_writter.lauc_errors)[:self.n_steps]
            self.mis_grouperrs = np.array(torch_writter.mis_grouperrs)[:, :self.n_steps]
            self.bce_grouperrs = np.array(torch_writter.bce_grouperrs)[:, :self.n_steps]
            self.auc_grouperrs = np.array(torch_writter.auc_grouperrs)[:, :self.n_steps]
            self.pauc_grouperrs = np.array(torch_writter.pauc_grouperrs)[:, :self.n_steps]
            self.lauc_grouperrs = np.array(torch_writter.lauc_grouperrs)[:, :self.n_steps]
            
        else:

            self.y_pred_proba = torch.stack(torch_writter.y_pred_proba).cpu().numpy()
            # self.loss = np.array(torch_writter.loss)
            # self.n_steps = len(self.loss)
            self.mis_errors = np.array(torch_writter.mis_errors)
            self.n_steps = len(self.mis_errors)
            self.bce_errors = np.array(torch_writter.bce_errors)
            self.auc_errors = np.array(torch_writter.auc_errors)
            self.pauc_errors = np.array(torch_writter.pauc_errors)
            self.lauc_errors = np.array(torch_writter.lauc_errors)
            self.mis_grouperrs = np.array(torch_writter.mis_grouperrs)
            self.bce_grouperrs = np.array(torch_writter.bce_grouperrs)
            self.auc_grouperrs = np.array(torch_writter.auc_grouperrs)
            self.pauc_grouperrs = np.array(torch_writter.pauc_grouperrs)
            self.lauc_grouperrs = np.array(torch_writter.lauc_grouperrs)

        self.agg_weights = np.arange(1, 1 + self.n_steps) ** 2
        self.agg_mis_errors = np.cumsum(self.mis_errors * self.agg_weights) / np.cumsum(self.agg_weights)
        self.agg_bce_errors = np.cumsum(self.bce_errors * self.agg_weights) / np.cumsum(self.agg_weights)
        self.agg_auc_errors = np.cumsum(self.auc_errors * self.agg_weights) / np.cumsum(self.agg_weights)
        self.agg_pauc_errors = np.cumsum(self.pauc_errors * self.agg_weights) / np.cumsum(self.agg_weights)
        self.agg_lauc_errors = np.cumsum(self.lauc_errors * self.agg_weights) / np.cumsum(self.agg_weights)
        self.agg_mis_grouperrs = np.cumsum(self.mis_grouperrs * self.agg_weights, axis=1) / np.cumsum(self.agg_weights)
        self.agg_bce_grouperrs = np.cumsum(self.bce_grouperrs * self.agg_weights, axis=1) / np.cumsum(self.agg_weights)
        self.agg_auc_grouperrs = np.cumsum(self.auc_grouperrs * self.agg_weights, axis=1) / np.cumsum(self.agg_weights)
        self.agg_pauc_grouperrs = np.cumsum(self.pauc_grouperrs * self.agg_weights, axis=1) / np.cumsum(self.agg_weights)
        self.agg_lauc_grouperrs = np.cumsum(self.lauc_grouperrs * self.agg_weights, axis=1) / np.cumsum(self.agg_weights)
        
        return
