import os
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from data import create_dataset, create_trn_val_tst_dataset
from models.utils.config import OptConfig
from models.early_fusion_multi_model import EarlyFusionMultiModel
from models.modality_miss_ts_model import ModalityMissTSModel
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def load_from_opt_record(file_path):
    opt_content = json.load(open(file_path, 'r'))
    opt = OptConfig()
    opt.load(opt_content)
    return opt

def load_model_from_checkpoints(module_class, teacher_path, cv):
    opt_path = os.path.join(teacher_path, 'train_opt.conf')
    opt = load_from_opt_record(opt_path)
    opt.isTrain = False                 
    opt.gpu_ids = [0]
    teacher_path_cv = os.path.join(teacher_path, str(cv))
    model = module_class(opt)
    model.cuda()
    model.load_networks_cv(teacher_path_cv)
    model.eval()
    return model

def load_dataset(dataset_name, cv):
    opt = fake_dataset_opt(dataset_name, cv)
    # print(dataset_name)
    # print(opt.__dict__)
    # input()
    dataset, val_dataset, tst_dataset = create_trn_val_tst_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    return dataset, val_dataset, tst_dataset


class fake_dataset_opt:
    def __init__(self, name, cv):
        self.dataset_mode = name
        self.batch_size = 256
        self.serial_batches = True
        self.num_threads = 0
        self.max_dataset_size = float('inf')
        self.cvNo = cv
        self.acoustic_ft_type = "IS10"
        self.visual_ft_type = "denseface"
        self.lexical_ft_type = "text"

def eval_10fold(model, val_iter):
    ans = {}
    for i, data in enumerate(val_iter):
        model.set_input(data)         
        model.test()
        preds = model.pred.argmax(dim=1).detach().cpu().numpy()
        for int2name, pred, label in zip(data['int2name'], preds, data['label']):
            label = label.cpu().numpy()
            for appendix in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
                ans[int2name + '_' + appendix] = pred == label
    return ans

def eval_mix(model, val_iter):
    ans = {}
    for i, data in enumerate(val_iter):
        model.set_input(data)         
        model.test()
        preds = model.pred.argmax(dim=1).detach().cpu().numpy()
        for int2name, pred, label, _type in zip(data['int2name'], preds, data['label'], data['miss_type']):
            label = label.cpu().numpy()
            ans[int2name + '_' + _type] = pred == label
    return ans

def filter_case(teacher_on_raw, teacher_on_miss, case_on_raw, case_on_miss):
    int2name_keys = list(teacher_on_raw.keys())
    ans = []
    for int2name in int2name_keys:
        if teacher_on_raw[int2name] and not teacher_on_miss[int2name] and case_on_miss[int2name]:
            ans.append(int2name)
    
    static = {}
    for part in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
        static[part] = 0
    
    for int2name in ans:
        part = int2name.split('_')[-1]
        static[part] += 1
    
    print(static)
    return ans      

if __name__ == '__main__':
    teacher_path = 'checkpoints/ef_AVL_Adnn512,256,128_Vlstm128_maxpool_Lcnn128_fusion256,128run1/'
    case_path = 'checkpoints/miss_ts_miss_mix_missmix_rate_0.5_Adnn512,256,128,256,128_lossBCE_kd1.0_temp2.0_ce0.5_mmd0.1_run1'
    # cv
    cv = 1
    
    teacher_model = load_model_from_checkpoints(EarlyFusionMultiModel, teacher_path, cv)
    case_model = load_model_from_checkpoints(ModalityMissTSModel, case_path, cv)
    _, raw_val, raw_tst = load_dataset('iemocap_10fold', cv)
    _, val_iter, tst_iter = load_dataset('iemocap_analysis', cv)
  
    # val
    ts_on_raw_val = eval_10fold(teacher_model, raw_val)
    ts_on_miss_val = eval_mix(teacher_model, val_iter)
    case_on_raw_val = eval_10fold(case_model, raw_val)
    case_on_mix_val = eval_mix(case_model, val_iter)
    # tst
    ts_on_raw_tst = eval_10fold(teacher_model, raw_tst)
    ts_on_miss_tst = eval_mix(teacher_model, tst_iter)
    case_on_raw_tst = eval_10fold(case_model, raw_tst)
    case_on_miss_tst = eval_mix(case_model, tst_iter)
    ans = filter_case(ts_on_raw_tst, ts_on_miss_tst, case_on_raw_tst, case_on_miss_tst)
    print(len(ans))
