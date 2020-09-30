# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 01:37:25 2020

@author: xuebing
"""


from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import shap
import timeit
import numpy as np
from sklearn.naive_bayes import GaussianNB
import os
from pathlib import Path
import scipy.stats as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet, SGDClassifier
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from scipy import interp
import math
from lightgbm import LGBMModel,LGBMClassifier
import glob
def create_dataset(path):
    df = pd.read_csv(path)
    data = df.values
    dataset = data[:,1:-1]
    labelset = data[:,-1]
    return dataset, labelset



def evaluation_per_class(y, pred_y):
    correct_label_list = [0, 0]
    total_label_list = [0, 0]
    for i in range(len(y)):
        if (y[i] == pred_y[i]):
            correct_label_list[y[i]] += 1
        total_label_list[y[i]] += 1

    TP = correct_label_list[1]
    TN = correct_label_list[0]
    FP = total_label_list[0] - correct_label_list[0]
    FN = total_label_list[1] - correct_label_list[1]
    # print "TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN
    sens = float(TP) / float(TP + FN) if float(TP + FN) != 0.0 else 0.0
    spec = float(TN) / float(TN + FP) if float(TN + FP) != 0.0 else 0.0
    PPV = float(TP) / float(TP + FP) if float(TP + FP) != 0.0 else 0.0
    f1 = 2.0 * PPV * sens / (PPV + sens) if float(PPV + sens) != 0.0 else 0.0
    acc = float(TP + TN) / float(total_label_list[0] + total_label_list[1])
    return sens, spec, PPV, f1, acc

def predict( clf_ind,dataset, labelset, train, test):
    result = np.zeros((1,9))
    train_dataset = dataset[train]
    train_labelset = labelset[train]
    test_dataset = dataset[test]
    test_labelset = labelset[test]
    X_resampled, y_resampled = RandomOverSampler().fit_sample(train_dataset,
                                                              train_labelset)
    train_dataset = X_resampled
    train_labelset = y_resampled
    if clf_ind == 0:
        clf = XGBClassifier(objective='binary:logistic', booster='gbtree')
    if clf_ind == 1:
        clf = LogisticRegression(solver='newton-cg', max_iter=1000, n_jobs=1)
    if clf_ind == 2:
        clf = RandomForestClassifier(n_estimators=300, max_depth=200, min_samples_split=4, max_features='sqrt',n_jobs=-1)
    if clf_ind == 5:
        linear_svc = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, shuffle=True,
                            learning_rate='optimal',n_jobs=-1)
        # This is the calibrated classifier which can give probabilistic classifier
        clf = CalibratedClassifierCV(linear_svc,
                                                method='sigmoid',
                                                # sigmoid will use Platt's scaling. Refer to documentation for other methods.
                                                cv=5)
    # train ML model
    clf.fit(train_dataset, train_labelset)
    y_test_pred = clf.predict_proba(train_dataset)[:, 1]
    fpr, tpr, thres = roc_curve(train_labelset, y_test_pred)
    auc_train = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(train_labelset, y_test_pred)
    prc_train = auc(rec, prec)
    # calculate AUROC, AUPRC
    y_test_pred = clf.predict_proba(test_dataset)[:, 1]
    fpr, tpr, thres = roc_curve(test_labelset, y_test_pred)
    auc_value = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(test_labelset, y_test_pred)
    auc_pr = auc(rec, prec)
    # report sensitivity, precision, f1, accuracy while fixing specificity~=0.95
    t = np.arange(0.0, 1.0, 0.01)
    diff = 1.0
    best_t = 0.5
    selected_metrics = []
    for j in range(t.shape[0]):
        dt = t[j] - 0.5
        sens, spec, PPV, f1, acc = evaluation_per_class(test_labelset.astype('int'),
                                                        np.round(y_test_pred - dt))
        if (abs(spec - 0.95) < diff):
            best_t = t[j]
            selected_metrics = [sens, spec, PPV, f1, acc]
            diff = abs(spec - 0.95)
    print('AUC:', auc_value, 'AUPRC:', auc_pr, 'metrics:', selected_metrics)
    result[0, 0:2] = [auc_train, prc_train]
    result[0, 2:4] = [auc_value, auc_pr]
    result[0, 4:] = selected_metrics
    return clf,result
def main(path, clf_ind):
    # path: path of precessed file (last column is outcome)
    # clf_ind: which machine learning algorithm to use
    cwd = os.getcwd()
    dataset, labelset = create_dataset(path)
    labelset=labelset.astype('int')
    n, d= dataset.shape
    column_names = ["index","train_auc", "train_prc", "test_roc", "test_prc", "sens", "spec","ppv", "f1", "acc"]
    results2 = pd.DataFrame(columns=column_names)
    results1 = np.zeros((n,25))
    ind = 0
    # 5 shuffles
    while ind < 25:
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        for i, (train, test) in enumerate(cv.split(dataset, labelset)):
            results2 = results2.append(pd.Series(), ignore_index=True)
            results2.iloc[ind,0] = ind
            clf, performance = predict(clf_ind, dataset, labelset, train, test)
            results2.iloc[ind, 1:] = performance
            results1[:,ind] = clf.predict_proba(dataset)[:, 1]
            ind += 1
    results2 = results2.append(pd.Series(), ignore_index=True)
    results2.iloc[ind, 0] = ind
    for col in range(1,10):
        a=results2.iloc[0:25,col]
        mean_col = np.mean(a)
        ci_col = st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))
        results2.iloc[ind, col] = str(round(mean_col,3)) +' (' + "%.3f" % round(ci_col[0], 3) + ',' + \
                                  "%.3f" % round(ci_col[1], 3) + ')'

        #save predictions of the dataset using each model (for plotting roc curve)
    np.savetxt(
        str(Path(cwd)) + "/results/predict_" + str(clf_ind) + "_" + str(os.path.basename(path)), results1, delimiter=",")
        # save performance metrics in each run 
    results2.to_csv(
        str(Path(cwd)) + "/results/performance_" + str(clf_ind) + "_" + str(os.path.basename(path)),
        index=False)
    explainer = shap.TreeExplainer(clf)
    dataset = pd.read_csv(path)
    shap_values = dataset.iloc[:,0:-1].copy()
    shap_values.iloc[1:-1] = explainer.shap_values(dataset.iloc[:,1:-1])
    shap_values.to_csv('shap_values_original.csv', index = False)

if __name__ == '__main__':
    main()
