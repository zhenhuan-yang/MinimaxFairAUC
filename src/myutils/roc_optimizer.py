'''
Tensor based roc operations for loss forwarding and backwarding
'''

import torch
import math

def mis(y_pred_proba, y):
    return (y != (y_pred_proba > 0.5).float()).float().mean().item()
    
def bce(y_pred_proba, y):
    bce = - (y * torch.log(y_pred_proba) + (1. - y) * torch.log(1. - y_pred_proba))    
    return bce.mean().item()

def lauc(score_neg, score_pos):
    # y_pos = torch.reshape(score_pos, (-1, 1))
    # y_neg = torch.reshape(score_neg, (-1, 1))
    # yy = y_pos - torch.transpose(y_neg, 0, 1) # [n_pos, n_neg] 2d-tensor
    yy = score_pos.reshape(-1, 1) - score_neg.reshape(1, -1)
    bce = - torch.log(torch.sigmoid(yy)) # [n_pos, n_neg] 2d-tensor
    return bce.mean().item()

def pauc(score_neg, score_pos, k=None, norm=True):
    diff = score_pos.reshape(-1, 1) - score_neg.reshape(1, -1)
    if k:
        diff = torch.topk(diff, k=k, largest=False)[0]
    if norm:
        return (diff > 0.).float().mean().item()
    else:
        return (diff > 0.).float().sum().item() / (len(score_neg) * len(score_pos))


def full_auc(score, y, approx_fun=torch.sigmoid):
    '''
    Tensor calculation
    Use transpose to simplify
    :param score:
    :param y:
    :return:
    '''
    score_pos, y_pos = score[y >= 0.5], y[y >= 0.5]
    score_neg, y_neg = score[y <= 0.5], y[y <= 0.5]  # same index

    num_pos = len(y_pos)
    num_neg = len(y_neg)
    if num_pos == 0 or num_neg == 0:
        # print('Only one class. Warning!')
        return torch.tensor(0., requires_grad=True), torch.tensor(0.)

    sc_neg = torch.reshape(score_neg, (-1, 1))
    sc_pos = torch.reshape(score_pos, (-1, 1))
    delta_sc = sc_neg - torch.transpose(sc_pos, 0, 1)
    # prod = delta_sc * .1
    # rel_auc = torch.mean(approx_fun(prod).log())
    prod = delta_sc * 2.
    rel_auc = torch.mean(approx_fun(prod))

    auc = torch.mean(torch.gt(prod, 0).float()) + 0.5 * torch.mean(torch.eq(prod, 0).float())

    # yy = torch.reshape(y, (-1, 1))
    # delta_y = yy - torch.transpose(yy, 0, 1)
    # n_diff = torch.sum(torch.ne(delta_y, 0).float())
    # if n_diff == 0:
    #     return torch.tensor([0.]), torch.tensor([0.])
    #
    # sc = torch.reshape(score, (-1, 1))
    # delta_sc = sc - torch.transpose(sc, 0, 1)
    # mask = torch.gt(delta_y, 0)
    # # We only select relevant instances to multiply
    # prod = delta_sc[mask] * delta_y[mask]
    # rel_auc = torch.mean(approx_fun(prod))
    # auc = torch.mean(torch.gt(prod, 0).float()) + 0.5 * torch.mean(torch.eq(prod, 0).float())

    return rel_auc, auc

def incomplete_auc(score, y, B=1024, approx_fun=torch.sigmoid):
    '''
    Faster calculation than transpose
    y = torch.tensor([1, 1, -1, -1])
    score = torch.tensor([0.9, 0.8, 0.2, 0.1])
    :param score:
    :param y:
    :param B:
    :return:
    '''
    score_pos, y_pos = score[y >= 0.5], y[y >= 0.5]
    score_neg, y_neg = score[y <= 0.5], y[y <= 0.5]

    num_pos = len(y_pos)
    num_neg = len(y_neg)
    if num_pos == 0 or num_neg == 0:
        return torch.tensor(0., requires_grad=True), torch.tensor(0.)

    # uniform random sample with replacement
    idx_pos = torch.randint(num_pos, (B,))
    idx_neg = torch.randint(num_neg, (B,))

    score_pos_B = score_pos[idx_pos]
    score_neg_B = score_neg[idx_neg]

    prod = 2. * (score_neg_B - score_pos_B)  # 2 coming from y_pos - y_neg
    rel_auc = torch.mean(approx_fun(prod))
    auc = torch.mean(torch.gt(prod, 0).float()) + 0.5 * torch.mean(torch.eq(prod, 0).float())

    return rel_auc, auc

def full_pauc(score, y, alpha=0., beta=.5, approx_fun=torch.sigmoid, reduction="mean"):
    '''
    Differentiable topk
    :param score:
    :param y:
    :param alpha:
    :param beta:
    :param approx_fun:
    :param reduction:
    :return:
    '''
    assert 0. <= beta <= 1. and 0. <= alpha <= beta, "Wrong range!"
    if alpha == beta:
        return torch.tensor(0., requires_grad=True), torch.tensor(0.)

    score_pos, y_pos = score[y >= 0.5], y[y >= 0.5]
    score_neg, y_neg = score[y <= 0.5], y[y <= 0.5] # same index

    num_pos = len(y_pos)
    num_neg = len(y_neg)
    if num_pos == 0 or num_neg == 0:
        return torch.tensor(0., requires_grad=True), torch.tensor(0.)
    # calculate the negative interval
    lower = math.floor(num_neg * alpha)
    upper = math.ceil(num_neg * beta)

    score_valid, valid_idx = torch.topk(score_neg, upper, dim=0) # score shape [-1, 1]
    if alpha != 0.:
        score_valid, valid_idx = torch.topk(score_valid, lower, dim=0, largest=False)
    # y_valid = y_neg[valid_idx] # same index

    # yy_valid = torch.reshape(y_valid, (-1, 1))
    # yy_pos = torch.reshape(y_pos, (-1, 1))
    # delta_y = yy_pos - torch.transpose(yy_valid, 0, 1)

    sc_valid = torch.reshape(score_valid, (-1, 1))
    sc_pos = torch.reshape(score_pos, (-1, 1))
    delta_sc = sc_valid - torch.transpose(sc_pos, 0, 1)
    prod = delta_sc * 2.
    if reduction == "mean":
        rel_pauc = torch.mean(approx_fun(prod))
        pauc = torch.mean(torch.gt(prod, 0.).float()) + 0.5 * torch.mean(torch.eq(prod, 0.).float())
    # for every negative take the mean of positive
    elif reduction == "pos_mean":
        rel_pauc = torch.mean(approx_fun(prod), dim=0)
        pauc = torch.mean(torch.gt(prod, 0.).float(), dim=0) + 0.5 * torch.mean(torch.eq(prod, 0.).float(), dim=0)
    else:
        raise NotImplemented
    return rel_pauc, pauc

def full_pauc_from_threshold(score, y, lambda_alpha, lambda_beta, alpha=0., beta=.5, approx_fun=torch.sigmoid):
    '''
    In fact it computes 1 - pAUC
    :param score:
    :param y:
    :param lambda_alpha: Tensor requires_grad = False/True
    :param lambda_beta: Tensor requires_grad = True
    :param alpha: required as zero
    :param beta:
    :param approx_fun:
    :return:
    '''
    assert 0. <= beta <= 1. and 0. <= alpha <= beta, "Wrong range!"
    if alpha == beta:
        return torch.tensor(0., requires_grad=True), torch.tensor(0.)

    score_pos, y_pos = score[y >= 0.5], y[y >= 0.5]
    score_neg, y_neg = score[y <= 0.5], y[y <= 0.5]

    num_pos = len(y_pos)
    num_neg = len(y_neg)
    if num_pos == 0 or num_neg == 0:
        return torch.tensor(0., requires_grad=True), torch.tensor(0.)

    lower = math.floor(num_neg * alpha)
    upper = math.ceil(num_neg * beta)

    sc_pos = torch.reshape(score_pos, (-1, 1))
    sc_neg = torch.reshape(score_neg, (-1, 1))
    delta_sc = sc_neg - torch.transpose(sc_pos, 0, 1)
    prod = delta_sc * 2.
    if alpha != 0.:
        inner_rel_pauc = torch.maximum(lambda_alpha - (approx_fun(prod).mean(1) - lambda_beta).maximum(torch.zeros(1)), torch.zeros(1)) # element-wise maximum
        inner_pauc = torch.maximum(lambda_alpha - (prod.gt(0).float().mean(1) - lambda_beta).maximum(torch.zeros(1)), torch.zeros(1)) # element-wise maximum
        rel_pauc = (num_neg - lower) / (upper - lower) * lambda_alpha + lambda_beta - torch.sum(inner_rel_pauc) / (upper - lower) # this definition affects the update of threshold
        pauc = (num_neg - lower) / (upper - lower) * lambda_alpha + lambda_beta - torch.sum(inner_pauc) / (upper - lower)
    else:
        inner_rel_pauc = (approx_fun(prod).mean(1) - lambda_beta).maximum(torch.zeros(1))  # element-wise maximum
        inner_pauc = (prod.gt(0).float().mean(1) - lambda_beta).maximum(torch.zeros(1))  # element-wise maximum
        rel_pauc = lambda_beta + torch.sum(inner_rel_pauc) / (upper - lower)  # this definition affects the update of threshold
        pauc = lambda_beta + torch.sum(inner_pauc) / (upper - lower)
    return rel_pauc, pauc

def topk_from_threshold(loss_0, loss_1, rho_gamma, ngamma):
    '''

    :param loss_avg: average loss over groups
    :param rho_gamma:
    :param ngamma: num_group * gamma
    :return:
    '''
    return rho_gamma + (torch.maximum(loss_0 - rho_gamma, torch.zeros(1)) + torch.maximum(loss_1 - rho_gamma, torch.zeros(1))) / ngamma

def cvar_from_threshold(loss, rho_gamma, gamma):
    '''

    :param loss_avg: average loss over groups
    :param rho_gamma:
    :param ngamma: num_group * gamma
    :return:
    '''
    return rho_gamma + torch.mean(torch.maximum(loss - rho_gamma, torch.zeros(1))) / gamma
