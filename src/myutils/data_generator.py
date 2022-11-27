'''
Generate synthetic dataset

'''

import os

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from myutils.data_utils import SummaryWritter

def create_gaussian(flip_rate=0.1, ratio=0.2, seed=42, n=[500, 100, 150, 10], m=[[0., 0.], [0.3, 0.3], [-0.2, 0.1], [0.2, 0.4]], 
                                c=[[[0.01, 0.], [0., 0.01]], [[0.01, 0.], [0., 0.01]], [[0.01, 0.], [0., 0.01]], [[0.01, 0.], [0., 0.01]]]):

    '''
    Input:
    n - number of sample
    m - mean
    c - covariance
    '''

    data_path = os.getcwd()
    if "utils" in data_path:
        data_path = os.path.dirname(data_path)
    data_path = os.path.join(data_path, "data/synthetic")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.random.seed(seed)

    n_00, n_01, n_10, n_11 = n[0], n[1], n[2], n[3]
    m_00, m_01, m_10, m_11 = m[0], m[1], m[2], m[3]
    c_00, c_01, c_10, c_11 = c[0], c[1], c[2], c[3]

    Y_00 = np.ones(n_00, dtype=np.float32) * 0
    Y_01 = np.ones(n_01, dtype=np.float32) * 1
    Y_10 = np.ones(n_10, dtype=np.float32) * 0
    Y_11 = np.ones(n_11, dtype=np.float32) * 1
    Z_00 = np.ones(n_00, dtype=np.float32) * 0
    Z_01 = np.ones(n_01, dtype=np.float32) * 0 
    Z_10 = np.ones(n_10, dtype=np.float32) * 1
    Z_11 = np.ones(n_11, dtype=np.float32) * 1

    X_00 = np.random.multivariate_normal(m_00, c_00, n_00)
    X_01 = np.random.multivariate_normal(m_01, c_01, n_01)
    X_10 = np.random.multivariate_normal(m_10, c_10, n_10)
    X_11 = np.random.multivariate_normal(m_11, c_11, n_11)

    flip_ind = np.random.choice(n_10, size=int(flip_rate * n_10), replace=False)
    Y_10[flip_ind] = 1.

    X = np.concatenate([X_00, X_01, X_10, X_11], axis=0, dtype=np.float32)
    Y = np.concatenate([Y_00, Y_01, Y_10, Y_11], axis=0, dtype=np.float32)
    Z = np.concatenate([Z_00, Z_01, Z_10, Z_11], axis=0, dtype=np.float32) 

    # clear output
    open(os.path.join(data_path, 'gaussian.p'), 'wb').close()
    with open(os.path.join(data_path, 'gaussian.p'), 'wb') as f:
        pkl.dump((X, Y, Z), f)

    # call the Writter class after loaded
    return X, Y, Z

def load_synthetic_data(group='gaussian', ratio=0.2, seed=42):
    
    data_path = os.getcwd()
    if "utils" in data_path:
        data_path = os.path.dirname(data_path)
    
    data_path = os.path.join(data_path, "data/synthetic/"+ group + '.p')
    with open(data_path, 'rb') as f:
        X, Y, Z = pkl.load(f)

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=ratio, random_state=seed, stratify=np.vstack((Y, Z)).T) 
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)  # the correct way to scandardize the test data
    
    train_writter = SummaryWritter(X_train, Y_train, Z_train)
    test_writter = SummaryWritter(X_test, Y_test, Z_test)

    X_train_list = train_writter.grouplabelsplitter(X_train, Y_train, Z_train)
    X_test_list = test_writter.grouplabelsplitter(X_test, Y_test, Z_test)

    return (X_train_list, train_writter), (X_test_list, test_writter)


GP_NAME_TO_FUNC = {"gaussian": create_gaussian}

def create_gp_by_name(gp_name, flip_rate=0.1, ratio=0.2, seed=42, **kwargs):

    create_data = GP_NAME_TO_FUNC[gp_name]
    if gp_name == 'gaussian':
        return create_data(flip_rate=flip_rate, ratio=ratio, seed=seed, **kwargs)

if __name__ == "__main__":

    gp_name = "gaussian"
    X, Y, Z = create_gp_by_name(gp_name, ratio=0.2, seed=42)