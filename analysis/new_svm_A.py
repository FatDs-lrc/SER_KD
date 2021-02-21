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
    data_root = 'checkpoints/analysis_recon_seq_zvl/{}'.format(cv)
    if not recon:
        trn_data = np.load(os.path.join(data_root, 'A_feat_trn.npy'))
        trn_label = np.load(os.path.join(data_root, 'trn_label.npy')).astype(np.int)
        val_data = np.load(os.path.join(data_root, 'A_feat_val.npy'))
        val_label = np.load(os.path.join(data_root, 'val_label.npy')).astype(np.int)
        tst_data = np.load(os.path.join(data_root, 'A_feat_test.npy'))
        tst_label = np.load(os.path.join(data_root, 'test_label.npy')).astype(np.int)
    else:
        trn_data = np.load(os.path.join(data_root, 'recon_A_feat_trn.npy'))
        trn_label = np.load(os.path.join(data_root, 'trn_label.npy')).astype(np.int)
        val_data = np.load(os.path.join(data_root, 'recon_A_feat_val.npy'))
        val_label = np.load(os.path.join(data_root, 'val_label.npy')).astype(np.int)
        tst_data = np.load(os.path.join(data_root, 'recon_A_feat_test.npy'))
        tst_label = np.load(os.path.join(data_root, 'test_label.npy')).astype(np.int)

    # trn_feat = standarize_data(trn_data)
    # val_feat = standarize_data(val_data)
    # tst_feat = standarize_data(tst_data)
    trn_feat = trn_data
    val_feat = val_data
    tst_feat = tst_data

    return trn_feat, trn_label, val_feat, val_label, tst_feat, tst_label

def svm_expr():
    # Load data from given feature path
    raw_feat_trn, raw_label_trn, raw_feat_val, raw_label_val, raw_feat_tst, raw_label_tst = load_trn_val_tst(False)
    recon_feat_trn, recon_label_trn, recon_feat_val, recon_label_val, recon_feat_tst, recon_label_tst = load_trn_val_tst(True)
    tuned_parameters = {'kernel': ['rbf', 'linear'], 'C': [0.1, 1.0], 'gamma': ['auto']}
    params = list(ParameterGrid(tuned_parameters))
    for param in params:
        print('------------------------------------------------------------')
        print("Current params: {}".format(param))
        raw_clf = SVC(probability=True, **param)
        raw_clf.fit(raw_feat_trn, raw_label_trn)
        y_true, y_pred = raw_label_tst, raw_clf.predict(raw_feat_tst)
        acc = accuracy_score(y_true, y_pred)
        uar = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred)
        print('RAW acc {:.4f} uar {:.4f} f1 {:.4f}'.format(acc, uar, f1))
        print('confusion matrix:\n{}'.format(cm))

        acc = uar = f1 = cm = y_true = y_pred = None

        recon_clf = SVC(probability=True, **param)
        recon_clf.fit(recon_feat_trn, recon_label_trn)
        y_true, y_pred = recon_label_tst, recon_clf.predict(recon_feat_tst)
        acc = accuracy_score(y_true, y_pred)
        uar = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred)
        print('Recon acc {:.4f} uar {:.4f} f1 {:.4f}'.format(acc, uar, f1))
        print('confusion matrix:\n{}'.format(cm))
        print('------------------------------------------------------------')

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
cv = int(sys.argv[1])
print('In cv {}'.format(cv))
svm_expr()
