'''
Pre-process datasets

------------------------- Adult ------------------------

attributes: "age", "workclass", "fnlwgt", "education", "education-num",
               "marital-status", "occupation", "relationship", "race", "sex",
               "capital-gain", "capital-loss", "hours-per-week",
               "native-country", "salary"

group: "sex"

label: "salary"

number of samples: 32561 + 16281

-------------------------- Bank -------------------------

attributes: 'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed', 'y'

group: "age"

label: "y"

number of samples: 41188

------------------------- Communities ------------------------

attributes:

group: "race" originally 'racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp'

label: "ViolentCrimesPerPop"

number of samples: 1994

------------------------- Compas ------------------------

attributes: 'id', 'name', 'first', 'last', 'compas_screening_date', 'sex', 'dob',
       'age', 'age_cat', 'race', 'juv_fel_count', 'decile_score',
       'juv_misd_count', 'juv_other_count', 'priors_count',
       'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number',
       'c_offense_date', 'c_arrest_date', 'c_days_from_compas',
       'c_charge_degree', 'c_charge_desc', 'is_recid', 'num_r_cases',
       'r_case_number', 'r_charge_degree', 'r_days_from_arrest',
       'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
       'is_violent_recid', 'num_vr_cases', 'vr_case_number',
       'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc',
       'v_type_of_assessment', 'v_decile_score', 'v_score_text',
       'v_screening_date', 'type_of_assessment', 'decile_score.1',
       'score_text', 'screening_date'

group: "race"

label: "is_recid"

number of samples: 11757

------------------------- Default ------------------------

attributes: 'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
       'default payment next month'

group: "sex"

label: "default payment next month"

nunmber of samples: 30000

------------------------- German ------------------------

attributes: 'check account', 'duration', 'credit history', 'purpose',
       'credit amount', 'savings/bonds', 'employed since', 'installment rate',
       'status and sex', 'other debtor/guarantor', 'residence since',
       'property', 'age', 'other plans', 'housing', 'existing credits', 'job',
       'number liable people', 'telephone', 'foreign worker',
       'credit decision'

group: "sex" originally "status and sex"

label: "credit decision"

number of samples: 1000

------------------------- Heart ------------------------

attributes: 'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
    'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
    'smoking', 'time', 'DEATH_EVENT'

group: "sex"

label: 'DEATH_EVENT'

number of samples: 299

------------------------- LSAC ------------------------

attributes: 'decile1b', 'decile3', 'ID', 'decile1', 'sex', 'race', 'cluster',
       'lsat', 'ugpa', 'zfygpa', 'DOB_yr', 'grad', 'zgpa', 'bar1', 'bar1_yr',
       'bar2', 'bar2_yr', 'fulltime', 'fam_inc', 'age', 'gender', 'parttime',
       'male', 'race1', 'race2', 'Dropout', 'other', 'asian', 'black', 'hisp',
       'pass_bar', 'bar', 'tier', 'index6040', 'indxgrp', 'indxgrp2'

group: "race"

label: 'pass_bar'

number of samples: 27478

'''
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import Dataset
from myutils.data_utils import SummaryWritter

DEF_TEST_SIZE = 0.20
BIG_TEST_SIZE = 0.40
TEST_SIZE_GERMAN = BIG_TEST_SIZE
TEST_SIZE_COMPAS = DEF_TEST_SIZE
TEST_SIZE_BANK = DEF_TEST_SIZE

class MyDataset(Dataset):
    '''
    For Dataloader
    '''
    def __init__(self, x, y, z, transform=None, target_transform=None):
        data = []
        for i in range(len(x)):
            data.append((x[i],y[i],z[i]))
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        feature, label, group = self.data[index]
        return feature, label, group

    def __len__(self):
        return len(self.data)

def load_adult_data(group="sex", ratio=0.2, seed=42, onehot=True):
    # The continuous variable fnlwgt represents final weight, which is the
    # number of units in the target population that the responding unit
    # represents.
    data_path = os.getcwd()
    if "utils" in data_path:
        data_path = os.path.dirname(data_path)
    data_path = os.path.join(data_path,"data/adult/processed_data.csv")
    df = pd.read_csv(data_path)

    if group == "sex":
        Z = np.array([0 if z == "Female" else 1 for z in df["sex"]], dtype=np.float32)
    elif group == "race":
        Z = np.array([0 if z == "non-White" else 1 for z in df["race"]], dtype=np.float32)
    elif group == "age":
        Z = np.array([0 if (z < 25) | (z > 60) else 1 for z in df["age"]], dtype=np.float32)
    else:
        raise ValueError("Invalid group name!")
 
    Y = np.array([1 if ">50K" in y else 0 for y in df['income']], dtype=np.float32)
    df = df[[a for a in df.columns if a not in ['income']]]

    col_quali = list(df.select_dtypes(include='O').columns)
    col_quanti = list(df.select_dtypes(include='int').columns)

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    if onehot:
        quali_encoder = OneHotEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali).toarray()
    else:
        quali_encoder = OrdinalEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali)

    X = np.concatenate([X_quanti, X_quali], axis=1, dtype=np.float32)

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=ratio, random_state=seed, stratify=np.vstack((Y, Z)).T)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # the correct way to scandardize the test data

    train_writter = SummaryWritter(X_train, Y_train, Z_train)
    test_writter = SummaryWritter(X_test, Y_test, Z_test)

    X_train_list = train_writter.grouplabelsplitter(X_train, Y_train, Z_train)
    X_test_list = test_writter.grouplabelsplitter(X_test, Y_test, Z_test)

    return (X_train_list, train_writter), (X_test_list, test_writter)

def load_bank_data(group="age", ratio=0.2, seed=42, onehot=True):
    # It is bank marketing data.
    # bank.csv 462K lines 450 Ko
    # bank-full 4M 614K lines 4.4 Mo
    # https://archive.ics.uci.edu/ml/datasets/bank+marketing
    data_path = os.getcwd()
    if "utils" in data_path:
        data_path = os.path.dirname(data_path)
    data_path = os.path.join(data_path, "data/bank/processed_data.csv")
    df = pd.read_csv(data_path)

    if group == "marital":
        Z = np.array([0 if z == "non-married" else 1 for z in df["marital"]], dtype=np.float32)
    elif group == "age":
        Z = np.array([0 if (z < 25) | (z > 60) else 1 for z in df["age"]], dtype=np.float32)
    else:
        raise ValueError("Invalid group name!")

    Y = np.array([1 if y == 'yes' else 0 for y in df['y']], dtype=np.float32)
    df = df[[a for a in df.columns if a not in ['y']]]

    col_quali = list(df.select_dtypes(include='O').columns)
    col_quanti = list(df.select_dtypes(include='int').columns)

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    if onehot:
        quali_encoder = OneHotEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali).toarray()
    else:
        quali_encoder = OrdinalEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali)

    X = np.concatenate([X_quanti, X_quali], axis=1, dtype=np.float32)

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

def load_communities_data(group="white", ratio=0.2, seed=42, onehot=True):

    # https://github.com/algowatchpenn/GerryFair/blob/master/dataset/communities.csv
    # https://github.com/amazon-research/minimax-fair/blob/main/src/clean_communities_data.py

    data_path = os.getcwd()
    if "utils" in data_path:
        data_path = os.path.dirname(data_path)
    data_path = os.path.join(data_path, "data/communities/processed_data.csv")
    df = pd.read_csv(data_path)

    if group == "race":
        Z = np.array([0 if z == "Black" else 1 for z in df["race"]], dtype=np.float32)
    else:
        raise ValueError("Invalid group name!")

    Y = np.array([1 if y > 0.7 else 0 for y in df['ViolentCrimesPerPop']], dtype=np.float32)
    df = df[[a for a in df.columns if a not in ['ViolentCrimesPerPop']]]

    col_quali = list(df.select_dtypes(include='O').columns)
    col_quanti = list(df.select_dtypes(include='float').columns)

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    if onehot:
        quali_encoder = OneHotEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali).toarray()
    else:
        quali_encoder = OrdinalEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali)

    X = np.concatenate([X_quanti, X_quali], axis=1, dtype=np.float32)

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=ratio, random_state=seed, stratify=np.vstack((Y, Z)).T)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_writter = SummaryWritter(X_train, Y_train, Z_train)
    test_writter = SummaryWritter(X_test, Y_test, Z_test)

    X_train_list = train_writter.grouplabelsplitter(X_train, Y_train, Z_train)
    X_test_list = test_writter.grouplabelsplitter(X_test, Y_test, Z_test)

    return (X_train_list, train_writter), (X_test_list, test_writter)

def load_compas_data(group="white", ratio=0.2, seed=42, onehot=True):
    # See https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm
    # Load the two-year data
    data_path = os.getcwd()
    if "utils" in data_path:
        data_path = os.path.dirname(data_path)
    data_path = os.path.join(data_path, "data/compas-analysis-master/processed_data.csv")
    df = pd.read_csv(data_path)

    if group == "race":
        Z = np.array([0 if z == "non-White" else 1 for z in df["race"]], dtype=np.float32)
    elif group == "sex":
        Z = np.array([0 if z == "Male" else 1 for z in df["sex"]], dtype=np.float32)
    else:
        raise ValueError("Invalid group name!")

    Y = np.array([1 if y == 1 else 0 for y in df['two_year_recid']], dtype=np.float32)
    df = df[[a for a in df.columns if a not in ['two_year_recid']]]

    col_quali = list(df.select_dtypes(include='O').columns)
    col_quanti = list(df.select_dtypes(include='int').columns)

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    if onehot:
        quali_encoder = OneHotEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali).toarray()
    else:
        quali_encoder = OrdinalEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali)

    X = np.concatenate([X_quanti, X_quali], axis=1, dtype=np.float32)

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=ratio, random_state=seed, stratify=np.vstack((Y, Z)).T)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_writter = SummaryWritter(X_train, Y_train, Z_train)
    test_writter = SummaryWritter(X_test, Y_test, Z_test)

    X_train_list = train_writter.grouplabelsplitter(X_train, Y_train, Z_train)
    X_test_list = test_writter.grouplabelsplitter(X_test, Y_test, Z_test)

    return (X_train_list, train_writter), (X_test_list, test_writter)

def load_default_data(group="sex", ratio=0.2, seed=42, onehot=True):

    data_path = os.getcwd()
    if "utils" in data_path:
        data_path = os.path.dirname(data_path)
    data_path = os.path.join(data_path, "data/default/processed_data.csv")
    df = pd.read_csv(data_path)

    if group == "sex":
        Z = np.array([0 if z == "Female" else 1 for z in df["SEX"]], dtype=np.float32)
    else:
        raise ValueError("Invalid group name!")

    Y = np.array([1 if y == 1 else 0 for y in df['default payment next month']], dtype=np.float32)
    df = df[[a for a in df.columns if a not in ['default payment next month']]]

    col_quanti = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2',
                  'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                  'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                  'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    col_quali = ['SEX', 'EDUCATION', 'MARRIAGE']

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    if onehot:
        quali_encoder = OneHotEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali).toarray()
    else:
        quali_encoder = OrdinalEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali)

    X = np.concatenate([X_quanti, X_quali], axis=1, dtype=np.float32)

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

def load_german_data(group="sex", ratio=0.2, seed=42, onehot=True):

    data_path = os.getcwd()
    if "utils" in data_path:
        data_path = os.path.dirname(data_path)
    data_path = os.path.join(data_path, "data/german/processed_data.csv")
    df = pd.read_csv(data_path)

    if group == "sex":
        Z = np.array([0 if z == "Female" else 1 for z in df["sex"]], dtype=np.float32)
    elif group == "age":
        Z = np.array([0 if z <= 25 else 1 for z in df['age']], dtype=np.float32) # bias towards young
    else:
        raise ValueError("Invalid group name!")

    Y = np.array([1 if y == 1 else 0 for y in df['class-label']], dtype=np.float32)
    df = df[[a for a in df.columns if a not in ['class-label']]]

    col_quali = list(df.select_dtypes(include='O').columns)
    col_quanti = list(df.select_dtypes(include='int').columns)

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    if onehot:
        quali_encoder = OneHotEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali).toarray()
    else:
        quali_encoder = OrdinalEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali)

    X = np.concatenate([X_quanti, X_quali], axis=1, dtype=np.float32)

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

def load_heart_data(group="sex", ratio=0.2, seed=42, onehot=True):

    data_path = os.getcwd()
    if "utils" in data_path:
        data_path = os.path.dirname(data_path)
    data_path = os.path.join(data_path, "data/heart/processed_data.csv")
    df = pd.read_csv(data_path)

    Y = df['DEATH_EVENT'].values.astype(np.float32)

    if group == "sex":
        Z = df['sex'].values.astype(np.float32)
    else:
        raise ValueError("Invalid group name!")

    X = df[df.columns[:-1]].values.astype(np.float32)

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=ratio, random_state=seed, stratify=np.vstack((Y, Z)).T)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_writter = SummaryWritter(X_train, Y_train, Z_train)
    test_writter = SummaryWritter(X_test, Y_test, Z_test)

    X_train_list = train_writter.grouplabelsplitter(X_train, Y_train, Z_train)
    X_test_list = test_writter.grouplabelsplitter(X_test, Y_test, Z_test)

    return (X_train_list, train_writter), (X_test_list, test_writter)

def load_lsac_data(group="sex", ratio=0.2, seed=42, onehot=True):
    '''
    Save data from
    https://github.com/lucweytingh/ARL-UvA/blob/master/data/preprocess_data/CreateLawSchoolDatasetFiles.ipynb
    Call preprocessing_lsac.ipynb to save data to .csv file
    '''
    data_path = os.getcwd()
    if "utils" in data_path:
        data_path = os.path.dirname(data_path)
    data_path = os.path.join(data_path, "data/law_school/processed_data.csv")
    df = pd.read_csv(data_path)

    if group == "sex":
        Z = np.array([1 if "Female" in z else 0 for z in df["sex"]], dtype=np.float32)
    elif group == "race":
        Z = np.array([1 if "non-White" in z else 0 for z in df["race"]], dtype=np.float32)
    else:
        raise ValueError("Invalid group name!")

    Y = np.array([1 if y == "Passed" else 0 for y in df['pass_bar']], dtype=np.float32)
    df = df[[a for a in df.columns if a not in ['pass_bar']]]

    col_quali = list(df.select_dtypes(include='O').columns)
    col_quanti = list(df.select_dtypes(include='float').columns)

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    if onehot:
        quali_encoder = OneHotEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali).toarray()
    else:
        quali_encoder = OrdinalEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali)

    X = np.concatenate([X_quanti, X_quali], axis=1, dtype=np.float32)

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=ratio, random_state=seed, stratify=np.vstack((Y, Z)).T)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_writter = SummaryWritter(X_train, Y_train, Z_train)
    test_writter = SummaryWritter(X_test, Y_test, Z_test)

    X_train_list = train_writter.grouplabelsplitter(X_train, Y_train, Z_train)
    X_test_list = test_writter.grouplabelsplitter(X_test, Y_test, Z_test)

    return (X_train_list, train_writter), (X_test_list, test_writter)

def load_mimiciii_data(group="race", ratio=0.4, seed=42, freq=5000):
    '''
    Preprocessing has been done
    Only loading is required
    '''

    data_path = os.getcwd()
    if "utils" in data_path:
        data_path = os.path.dirname(data_path)
    
    data_path = os.path.join(data_path, "data/mimiciii/processed_data_group_{}_ratio_{}_seed_{}_freq_{}.pkl".format(group, ratio, seed, freq))

    with open(data_path, "rb") as f:
        (X_train_list, Y_train, Z_train), (X_test_list, Y_test, Z_test) = pickle.load(f)

    train_writter = SummaryWritter([], Y_train, Z_train)
    test_writter = SummaryWritter([], Y_test, Z_test)

    train_writter.n_features = freq
    test_writter.n_features = freq

    return (X_train_list, train_writter), (X_test_list, test_writter)

def load_oulad_data(group="sex", ratio=0.2, seed=42, onehot=True):
    '''
    Save data from
    https://github.com/SergioSim/OULAD/blob/master/tools/load_oulad.py
    Call preprocessing_oulad.ipynb to save data to .csv file
    '''
    data_path = os.getcwd()
    if "utils" in data_path:
        data_path = os.path.dirname(data_path)
    data_path = os.path.join(data_path, "data/law_school/processed_data.csv")
    df = pd.read_csv(data_path)

    if group == "sex":
        Z = np.array([1 if "Female" in z else 0 for z in df["sex"]], dtype=np.float32)
    elif group == "race":
        Z = np.array([1 if "non-White" in z else 0 for z in df["race"]], dtype=np.float32)
    else:
        raise ValueError("Invalid group name!")

    Y = np.array([1 if y == "Passed" else 0 for y in df['pass_bar']], dtype=np.float32)
    df = df[[a for a in df.columns if a not in ['pass_bar']]]

    col_quali = list(df.select_dtypes(include='O').columns)
    col_quanti = list(df.select_dtypes(include='float').columns)

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    if onehot:
        quali_encoder = OneHotEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali).toarray()
    else:
        quali_encoder = OrdinalEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali)

    X = np.concatenate([X_quanti, X_quali], axis=1, dtype=np.float32)

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=ratio, random_state=seed, stratify=np.vstack((Y, Z)).T)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_writter = SummaryWritter(X_train, Y_train, Z_train)
    test_writter = SummaryWritter(X_test, Y_test, Z_test)

    X_train_list = train_writter.grouplabelsplitter(X_train, Y_train, Z_train)
    X_test_list = test_writter.grouplabelsplitter(X_test, Y_test, Z_test)

    return (X_train_list, train_writter), (X_test_list, test_writter)

def load_ricci_data(group="sex", ratio=0.2, seed=42, onehot=True):

    data_path = os.getcwd()
    if "utils" in data_path:
        data_path = os.path.dirname(data_path)
    data_path = os.path.join(data_path, "data/ricci/processed_data.csv")
    df = pd.read_csv(data_path)

    if group == "race":
        Z = np.array([1 if z == "non-White" else 0 for z in df["Race"]], dtype=np.float32)
    else:
        raise ValueError("Invalid group name!")

    Y = np.array([1 if y >= 70 else 0 for y in df['Combine']], dtype=np.float32)
    df = df[[a for a in df.columns if a not in ['Combine']]]

    col_quali = list(df.select_dtypes(include='O').columns)
    col_quanti = list(df.select_dtypes(include='float').columns)

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    if onehot:
        quali_encoder = OneHotEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali).toarray()
    else:
        quali_encoder = OrdinalEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali)

    X = np.concatenate([X_quanti, X_quali], axis=1, dtype=np.float32)

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=ratio, random_state=seed, stratify=np.vstack((Y, Z)).T)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_writter = SummaryWritter(X_train, Y_train, Z_train)
    test_writter = SummaryWritter(X_test, Y_test, Z_test)

    X_train_list = train_writter.grouplabelsplitter(X_train, Y_train, Z_train)
    X_test_list = test_writter.grouplabelsplitter(X_test, Y_test, Z_test)

    return (X_train_list, train_writter), (X_test_list, test_writter)

def load_student_data(group="sex", ratio=0.2, seed=42, onehot=True):

    data_path = os.getcwd()
    if "utils" in data_path:
        data_path = os.path.dirname(data_path)
    data_path = os.path.join(data_path, "data/student/processed_data.csv")
    df = pd.read_csv(data_path)

    if group == "sex":
        Z = np.array([1 if z == "Female" else 0 for z in df["sex"]], dtype=np.float32)
    elif group == "age":
        Z = np.array([1 if z >= 18 else 0 for z in df["age"]], dtype=np.float32)
    else:
        raise ValueError("Invalid group name!")

    Y = np.array([1 if y >= 10 else 0 for y in df['G3']], dtype=np.float32)
    df = df[[a for a in df.columns if a not in ['G3']]]

    col_quali = list(df.select_dtypes(include='O').columns)
    col_quanti = list(df.select_dtypes(include='int').columns)

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    if onehot:
        quali_encoder = OneHotEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali).toarray()
    else:
        quali_encoder = OrdinalEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali)

    X = np.concatenate([X_quanti, X_quali], axis=1, dtype=np.float32)

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=ratio, random_state=seed, stratify=np.vstack((Y, Z)).T)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_writter = SummaryWritter(X_train, Y_train, Z_train)
    test_writter = SummaryWritter(X_test, Y_test, Z_test)

    X_train_list = train_writter.grouplabelsplitter(X_train, Y_train, Z_train)
    X_test_list = test_writter.grouplabelsplitter(X_test, Y_test, Z_test)

    return (X_train_list, train_writter), (X_test_list, test_writter)

def custom_split(X, y, z, test_size=0.2, random_state=1):

    # make sure train and val both have all the groups
    n_groups = len(np.unique(z))
    groups_ids = [np.where(z == i)[0] for i in range(n_groups)]
    n_labels = len(np.unique(y))
    labels_ids = [np.where(y == j)[0] for j in range(n_labels)]
    # setup a [group, label] indices list
    grouplabels_ids = [[labels_ids[j][z[labels_ids[j]] == i] for j in range(n_labels)] for i in range(n_groups)]
    # split
    train_ids = set()
    val_ids = set()
    for group in grouplabels_ids:
        for grouplabel_ids in group:
            temp_train_grouplabel_ids, temp_val_grouplabel_ids = train_test_split(grouplabel_ids, test_size=test_size,
                                                                        random_state=random_state)
            assert len(temp_train_grouplabel_ids) >= 1
            assert len(temp_val_grouplabel_ids) >= 1
            train_ids.update(temp_train_grouplabel_ids)
            val_ids.update(temp_val_grouplabel_ids)
    # split train and validation
    X_train, y_train, z_train = X[list(train_ids)], y[list(train_ids)], z[list(train_ids)]
    X_val, y_val, z_val = X[list(val_ids)], y[list(val_ids)], z[list(val_ids)]

    return X_train, X_val, y_train, y_val, z_train, z_val

DB_NAME_TO_FUNC = {"adult": load_adult_data, "bank": load_bank_data, "communities": load_communities_data,
                   "compas": load_compas_data, "default": load_default_data, "german": load_german_data,
                   "heart": load_heart_data, "lsac": load_lsac_data, "mimiciii": load_mimiciii_data,
                   "ricci": load_ricci_data, "student": load_student_data}

def load_db_by_name(db_name, gp_name="sex", ratio=0.4, seed=42, onehot=True, freq=5000):

    load_data = DB_NAME_TO_FUNC[db_name]
    if "mimic" in db_name:
        return load_data(gp_name, ratio=ratio, seed=seed, freq=freq)
    else:
        return load_data(gp_name, ratio=ratio, seed=seed, onehot=onehot)

if __name__ == "__main__":
    import datetime
    import torch
    time_1 = datetime.datetime.now()
    db_name = "mimiciii"
    gp_name = "sex"
    (X_train, train_writter), (X_test, test_writter) = load_db_by_name(db_name, gp_name, ratio=0.4)
    time_2 = datetime.datetime.now()
    print('Loading time: {}'.format(time_2 - time_1))
    
    for g in range(train_writter.n_groups):
        values = X_train[2*g].data
        indices = np.vstack((X_train[2*g].row, X_train[2*g].col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = X_train[2*g].shape
        X_train_neg = torch.sparse.FloatTensor(i, v, torch.Size(shape))

