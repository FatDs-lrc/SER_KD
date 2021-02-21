from __future__ import print_function

import os
import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn import preprocessing

def standarize_data(data):
    return preprocessing.scale(data)

def load_trn_val_tst(recon=False):
    data_root = 'checkpoints/analysis_recon_seq_avz/10'
    if not recon:
        trn_data = np.load(os.path.join(data_root, 'L_feat_trn.npy'))
        trn_label = np.load(os.path.join(data_root, 'trn_label.npy')).astype(np.int)
        val_data = np.load(os.path.join(data_root, 'L_feat_val.npy'))
        val_label = np.load(os.path.join(data_root, 'val_label.npy')).astype(np.int)
        tst_data = np.load(os.path.join(data_root, 'L_feat_test.npy'))
        tst_label = np.load(os.path.join(data_root, 'test_label.npy')).astype(np.int)
    else:
        trn_data = np.load(os.path.join(data_root, 'recon_L_feat_trn.npy'))
        trn_label = np.load(os.path.join(data_root, 'trn_label.npy')).astype(np.int)
        val_data = np.load(os.path.join(data_root, 'recon_L_feat_val.npy'))
        val_label = np.load(os.path.join(data_root, 'val_label.npy')).astype(np.int)
        tst_data = np.load(os.path.join(data_root, 'recon_L_feat_test.npy'))
        tst_label = np.load(os.path.join(data_root, 'test_label.npy')).astype(np.int)

    # trn_feat = standarize_data(trn_data)
    # val_feat = standarize_data(val_data)
    # tst_feat = standarize_data(tst_data)
    trn_feat = trn_data
    val_feat = val_data
    tst_feat = tst_data

    return trn_feat, trn_label, val_feat, val_label, tst_feat, tst_label

def svm_expr(recon=False):
    print('Recon : {}'.format(recon))
    # Load data from given feature path
    feat_trn, label_trn, feat_val, label_val, feat_tst, label_tst = load_trn_val_tst(recon)
    # Set the parameters by cross-validation
    # tuned_parameters = [{'kernel': ['rbf', 'sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [0.1, 1, 10, 100, 1000], 'class_weight':['balanced', {1:2, 2:1, 3:4}, {1:4, 2:1, 3:10}, {1:6, 2:1, 3:20}]},
    #                  {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000], 'class_weight':['balanced', {1:2, 2:1, 3:4}, {1:4, 2:1, 3:10}, {1:6, 2:1, 3:20}]}]
    # tuned_parameters = {'kernel':['rbf'], 'gamma':[0.001], 'C': [10], 'class_weight':['balanced', {'1':2, '2':1, '3':4}, {'1':4, '2':1, '3':10}, {'1':6, '2':1, '3':20}]}
    # tuned_parameters = {'kernel':['rbf'], 'gamma':[0.001], 'C': [10], 'class_weight':[{1:6, 2:1, 3:20}]}
    # tuned_parameters = {'kernel': ['rbf']}
    # tuned_parameters = {'kernel':['rbf'], 'C':[10], 'gamma':[1e-3], 'class_weight':['balanced', {1:2, 2:1, 3:4}, {1:2, 2:1, 3:10}, {1:4, 2:1, 3:12}]}
    tuned_parameters = {'kernel': ['rbf', 'sigmoid', 'poly'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [0.01, 0.1, 1, 10, 100]}
    params = list(ParameterGrid(tuned_parameters))
    best_clf = None
    uar_max = 0.0
    best_param = None
    for param in params:
        print("\n# Tuning hyper-parameters:")
        print("Current params: {}".format(param))
        clf = SVC(probability=True, **param)
        clf.fit(feat_trn, label_trn)
        val_true, val_pred = label_val, clf.predict(feat_val)
        # f1 = f1_score(val_true, , average='macro')
        acc = accuracy_score(val_true, val_pred)
        uar = recall_score(val_true, val_pred, average='macro')
        f1 = f1_score(val_true, val_pred, average='macro')
        cm = confusion_matrix(val_true, val_pred)
        print('On VAL Param: {} \nacc {:.4f} uar {:.4f} f1 {:.4f}'.format(param, acc, uar, f1))
        # if uar > uar_max:
        #     uar_max = uar
        #     best_param = param
        #     best_clf = clf

        

        y_true, y_pred = label_tst, clf.predict(feat_tst)
        acc = accuracy_score(y_true, y_pred)
        uar = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred)
        print('On TST: Param: {} \nacc {:.4f} uar {:.4f} f1 {:.4f}'.format(best_param, acc, uar, f1))
        if uar > uar_max:
            uar_max = uar
            best_param = param
            best_clf = clf

    print("Best parameters set found on evaluation set:{}\n\n".format(best_param))
    if full_training:
        print("The model is trained on the full development set.")
        X = np.concantenate(feat_trn, feat_val)
        y = np.concantenate(label_trn, label_val)
        best_clf = SVC(probability=True, **best_param)
        best_clf.fit(X)

    print("Detailed classification report:\n")

    y_true, y_pred = label_tst, best_clf.predict(feat_tst)
    acc = accuracy_score(y_true, y_pred)
    uar = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)

    print('On TST: Param: {} \nacc {:.4f} uar {:.4f} f1 {:.4f}'.format(best_param, acc, uar, f1))
    print(classification_report(y_true, y_pred))
    print()
    print('Confusion matrix:\n{}'.format(confusion_matrix(y_true, y_pred)))
    pwd = os.path.dirname(__file__)
    # write to txt
    f = open(os.path.join(pwd, 'svm_report', 'recon_{}.txt'.format(recon)), 'w')
    f.write(classification_report(y_true, y_pred) + '\n')
    f.write('Confusion matrix:\n{}'.format(confusion_matrix(y_true, y_pred)))
    f.write('\n')
    f.close()

def hh(recon):
    feat_trn, label_trn, feat_val, label_val, feat_tst, label_tst = load_trn_val_tst(recon)
    # best_param = {'C': 0.1, 'gamma': 0.0001, 'kernel': 'sigmoid'}
    best_param = {'C': 0.01, 'gamma': 0.0001, 'kernel': 'rbf'}
    clf = SVC(probability=True, **best_param) 
    clf.fit(feat_trn, label_trn)
    y_true, y_pred = label_tst, clf.predict(feat_tst)
    acc = accuracy_score(y_true, y_pred)
    uar = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    print('On TST: Param: {} \nacc {:.4f} uar {:.4f} f1 {:.4f}'.format(best_param, acc, uar, f1))
    print(classification_report(y_true, y_pred))
    print()
    print('Confusion matrix:\n{}'.format(confusion_matrix(y_true, y_pred)))


import sys
full_training = False
recon = sys.argv[1]
recon = True if recon == 'recon' else False
svm_expr(recon)
# hh(recon)

'''
raw:
On TST: Param: {'C': 0.01, 'gamma': 0.001, 'kernel': 'rbf'} 
acc 0.7020 uar 0.7204 f1 0.7080
Confusion matrix:
[[ 67   3  10   2]
 [  4 112  21   9]
 [ 17  40 135  21]
 [  8  11  20  77]]


recon:
On TST: Param: {'C': 0.1, 'gamma': 0.001, 'kernel': 'rbf'} 
acc 0.6732 uar 0.6967 f1 0.6815
Confusion matrix:
[[ 61   6  10   5]
 [ 10 111  18   7]
 [ 14  37 119  43]
 [  2   6  24  84]]
'''
'''
seq
CV1:
raw:
On TST: Param: {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'} 
acc 0.6174 uar 0.6279 f1 0.6078
[[106   9  26   6]
 [  4 102  13  13]
 [ 23  35  71  42]
 [  4  17  10  47]]

recon:
On TST: Param: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'} 
acc 0.5644 uar 0.6137 f1 0.543
[[112  15  16   4]
 [ 30  92   2   8]
 [ 18  43  30  80]
 [  0  11   3  64]]

 CV2:
 raw:
 On TST: Param: {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'} 
 acc 0.6715 uar 0.7017 f1 0.6811
 [[ 67   4  10   1]
 [  5 107  21  13]
 [ 20  47 119  27]
 [  8  11  16  81]]

 recon:
 On TST: Param: {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'} 
 acc 0.5260 uar 0.5802 f1 0.5298
 [[ 53   3  14  12]
 [ 26  61  26  33]
 [ 21  19  73 100]
 [  4   1   5 106]]
 
 cv4:
 raw:
 On TST: Param: {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'} 
 acc 0.6827 uar 0.7144 f1 0.6837
 [[ 57   4   6   3]
 [  5 123  18  15]
 [ 19  51 131  26]
 [  5  11   9  59]]

 recon:
 On TST: Param: {'C': 100, 'gamma': 0.01, 'kernel': 'poly'} 
 acc 0.5572 uar 0.5819 f1 0.5682
[[ 49  20   1   0]
 [ 19 118  24   0]
 [ 14  98  95  20]
 [  2  15  27  40]]

 cv8:
 raw:
 On TST: Param: {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'} 
 acc 0.6382 uar 0.6538 f1 0.6490
 [[80  9 29  4]
 [ 2 89 18  9]
 [17 54 98 13]
 [ 4  7 16 54]]

 reconL
 On TST: Param: {'C': 100, 'gamma': 0.01, 'kernel': 'poly'} 
 acc 0.5507 uar 0.5343 f1 0.5476
 [[ 78  27  17   0]
 [ 27  53  34   4]
 [ 18  45 110   9]
 [  3   0  42  36]]

'''