# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 01:37:25 2020

@author: xuebing
"""



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator
from sklearn.pipeline import FeatureUnion, make_pipeline
def one_hot_encoding(combined,ind, variables):
# one hot encoding
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    indicator = MissingIndicator(missing_values=np.nan, features='all')
    transformer = FeatureUnion(transformer_list=[('features', imp), ('indicators', indicator)])
    scaler = StandardScaler()

    for i, v in enumerate(combined.columns.values):
        if v in variables:
            combined = pd.concat([combined, pd.get_dummies(combined[v], prefix=v, dummy_na=ind, drop_first=True)], axis=1)
            combined.drop([v], axis=1, inplace=True)
        elif not v=='PatientID':
            x_scaled = scaler.fit_transform(combined[v].values)
            combined.v[:] = x_scaled
            temp_file_transform = transformer.fit_transform(combined[v].values)
            combined.v[:] = temp_file_transform[:,0]
            dummy_name = 'dummy' + v
            combined[dummy_name] = temp_file_transform[:,1]
    return combined

def main(variables, path_inop, path_preop, outcome):
    # path_preop: file of preoperative features with the first column being patientid ('PatientID'), and missing fields with nan
    # path_inop : file of intraoperative features with the first column being patientid ('PatientID') and missing fields with nan. If not familiar with
    #               the extraction of statistical features, can use package: https://tsfresh.readthedocs.io/en/latest/
    # outcome: the column name of outcome
    # example of variables (list of column names of categorical features):
    # variables = ['LVEF', 'Outpatient_Insulin', 'ASA', 'ASA_EM', 'VALVULAR_DISEASE', 'CHF_Diastolic_Function',
    #              'Dialysis_History',
    #              'SEX', 'CKD', 'CCI', 'RACE', 'AFIB', 'SMOKING_EVER', 'FUNCTIONAL_CAPACITY', 'Surg_Type', 'Anesthesia_Type',
    #              'pdel']

    inop = pd.read_csv(path_inop)
    preop = pd.read_csv(path_preop)
    combined = preop.join(inop.set_index('PatientID'), on='PatientID', how='inner')
    df2 = combined.pop(outcome)
    combined = one_hot_encoding(combined, 1, variables)
    combined[outcome] = df2
    fname=outcome + '_processed_file.csv'
    combined.to_csv(fname, index = False)



