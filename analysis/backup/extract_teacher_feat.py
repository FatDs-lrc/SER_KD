import os
import time
import json
import numpy as np
from tqdm import tqdm
from data import create_dataset, create_trn_val_tst_dataset
from models.utils.config import OptConfig
from models.networks.tools import MultiLayerFeatureExtractor
from models.early_fusion_multi_model import EarlyFusionMultiModel
from models.multi_fusion_multi_model import MultiFusionMultiModel
from models.new_translation_model import NewTranslationModel
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract(model, val_iter, save_path, phase='val', modality='A'):
    model.eval()
    total_feat = []
    total_label = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        # feat = getattr(model, 'embd_{}'.format(modality)).detach().cpu().numpy()
        # feat = extractor.extract()[0].cpu().numpy()
        feat = getattr(model, 'feat_{}'.format(modality)).detach().cpu().numpy()
        label = data['label']
        total_feat.append(feat)
        total_label.append(label)
    
    # calculate metrics
    total_feat = np.vstack(total_feat)
    total_label = np.concatenate(total_label)
    
    # save model_feat
    np.save(os.path.join(save_path, '{}.npy'.format(phase)), total_feat)
    np.save(os.path.join(save_path, '{}_label.npy'.format(phase)), total_label)

def extract_feat(model, val_iter, attr, save_path, cv, phase='val'):
    cv = str(cv)
    model.eval()
    total_feat = []
    total_label = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        # feat = getattr(model, 'embd_{}'.format(modality)).detach().cpu().numpy()
        # feat = extractor.extract()[0].cpu().numpy()
        feat = getattr(model, attr).detach().cpu().numpy()
        label = data['label']
        total_feat.append(feat)
        total_label.append(label)
    
    # calculate metrics
    total_feat = np.vstack(total_feat)
    total_label = np.concatenate(total_label)
    
    save_dir = os.path.join(save_path, cv)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # save model_feat
    np.save(os.path.join(save_dir, '{}.npy'.format(phase)), total_feat)
    np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

def extract_all(model, val_iter, save_path, cv, phase='val'):
    model.eval()
    A_feats = []
    V_feats = []
    L_feats = []
    fusion_feats = []
    total_label = []
    total_int2name = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        feat_A = getattr(model, 'feat_A').detach().cpu().numpy()
        feat_V = getattr(model, 'feat_V').detach().cpu().numpy()
        feat_L = getattr(model, 'feat_L').detach().cpu().numpy()
        # feat_fusion = extractor.extract()[0].cpu().numpy()
        feat_fusion = getattr(model, 'ef_fusion_feat').detach().cpu().numpy()
        label = data['label']
        # int2name = data['int2name']
        A_feats.append(feat_A)
        V_feats.append(feat_V)
        L_feats.append(feat_L)
        fusion_feats.append(feat_fusion)
        total_label.append(label)
        # total_int2name.append(int2name)
    
    # calculate metrics
    A_feats = np.vstack(A_feats)
    V_feats = np.vstack(V_feats)
    L_feats = np.vstack(L_feats)
    fusion_feats = np.vstack(fusion_feats)
    total_label = np.concatenate(total_label)
    # total_int2name = np.concatenate(total_int2name)
    
    # save model_feat
    feat_save_path = os.path.join(save_path, '{}', str(cv), '{}.npy'.format(phase))
    for modality in ['A', 'V', 'L', 'fusion']:
        if not os.path.exists(os.path.dirname(feat_save_path.format(modality))):
            os.makedirs(os.path.dirname(feat_save_path.format(modality)))
        
    np.save(feat_save_path.format('A'), A_feats)
    np.save(feat_save_path.format('V'), V_feats)
    np.save(feat_save_path.format('L'), L_feats)
    np.save(feat_save_path.format('fusion'), fusion_feats)
    # save target
    target_save_path = os.path.join(save_path, 'target', str(cv))
    if not os.path.exists(target_save_path):
        os.makedirs(target_save_path)

    np.save(os.path.join(target_save_path, '{}_label.npy'.format(phase)), total_label)
    # np.save(os.path.join(target_save_path, '{}_int2name.npy'.format(phase)), total_int2name)

def extract_all_mix(model, val_iter, save_path, cv, phase='val'):
    print(cv, phase)
    model.eval()
    A_feats = []
    V_feats = []
    L_feats = []
    total_label = []
    total_int2name = []
    total_miss_type = []
    total_miss_index = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        feat_A = getattr(model, 'feat_A').detach().cpu().numpy()
        feat_V = getattr(model, 'feat_V').detach().cpu().numpy()
        feat_L = getattr(model, 'feat_L').detach().cpu().numpy()
        label = data['label']
        int2name = data['int2name']
        for a, v, l, _label, _int2name in zip(feat_A, feat_V, feat_L, label, int2name):
            a = a.reshape(1, -1)
            v = v.reshape(1, -1)
            l = l.reshape(1, -1)
            # A + Z + Z
            A_feats.append(a)
            V_feats.append(np.zeros(v.shape))
            L_feats.append(np.zeros(l.shape))
            total_label.append(_label)
            total_int2name.append(_int2name)
            total_miss_type.append('azz')
            total_miss_index.append([1, 0, 0])
            # Z + V + Z
            A_feats.append(np.zeros(a.shape))
            V_feats.append(v)
            L_feats.append(np.zeros(l.shape))
            total_label.append(_label)
            total_int2name.append(_int2name)
            total_miss_type.append('zvz')
            total_miss_index.append([0, 1, 0])
            # Z + Z + L
            A_feats.append(np.zeros(a.shape))
            V_feats.append(np.zeros(v.shape))
            L_feats.append(l)
            total_label.append(_label)
            total_int2name.append(_int2name)
            total_miss_type.append('zzl')
            total_miss_index.append([0, 0, 1])
            # A + V + Z
            A_feats.append(a)
            V_feats.append(v)
            L_feats.append(np.zeros(l.shape))
            total_label.append(_label)
            total_int2name.append(_int2name)
            total_miss_type.append('avz')
            total_miss_index.append([1, 1, 0])
            # A + Z + L
            A_feats.append(a)
            V_feats.append(np.zeros(v.shape))
            L_feats.append(l)
            total_label.append(_label)
            total_int2name.append(_int2name)
            total_miss_type.append('azl')
            total_miss_index.append([1, 0, 1])
            # Z + V + L
            A_feats.append(np.zeros(a.shape))
            V_feats.append(v)
            L_feats.append(l)
            total_label.append(_label)
            total_int2name.append(_int2name)
            total_miss_type.append('zvl')
            total_miss_index.append([0, 1, 1])
                
    # calculate metrics
    A_feats = np.vstack(A_feats)
    V_feats = np.vstack(V_feats)
    L_feats = np.vstack(L_feats)
    fusion_feats = np.concatenate([A_feats, V_feats, L_feats], axis=1)
    print(fusion_feats.shape)
    input()
    total_label = np.array(total_label)
    total_int2name = np.array(total_int2name)
    total_miss_type = np.array(total_miss_type)
    total_miss_index = np.array(total_miss_index)

    # save model_feat
    feat_save_path = os.path.join(save_path, '{}', str(cv), '{}.npy'.format(phase))
    for modality in ['A', 'V', 'L', 'fusion']:
        if not os.path.exists(os.path.dirname(feat_save_path.format(modality))):
            os.makedirs(os.path.dirname(feat_save_path.format(modality)))
        
    np.save(feat_save_path.format('A'), A_feats)
    np.save(feat_save_path.format('V'), V_feats)
    np.save(feat_save_path.format('L'), L_feats)
    np.save(feat_save_path.format('fusion'), fusion_feats)
    # save target
    target_save_path = os.path.join(save_path, 'target', str(cv))
    if not os.path.exists(target_save_path):
        os.makedirs(target_save_path)

    np.save(os.path.join(target_save_path, '{}_label.npy'.format(phase)), total_label)
    np.save(os.path.join(target_save_path, '{}_int2name.npy'.format(phase)), total_int2name)
    np.save(os.path.join(target_save_path, '{}_type.npy'.format(phase)), total_miss_type)
    np.save(os.path.join(target_save_path, '{}_miss_index.npy'.format(phase)), total_miss_index)


def load_from_opt_record(file_path):
    opt_content = json.load(open(file_path, 'r'))
    opt = OptConfig()
    opt.load(opt_content)
    return opt


if __name__ == '__main__':
    teacher_path = 'checkpoints/ef_AVL_Adnn512,256,128_Vlstm128_maxpool_Lcnn128_fusion256,128run1/'
    # teacher_path = 'checkpoints/new_0416_simpleAE_ce_t1.0_f1.0_mse0.1_cycle0.1_run1'
    opt_path = os.path.join(teacher_path, 'train_opt.conf')
    opt = load_from_opt_record(opt_path)
    opt.isTrain = False                             # teacher model should be in test mode
    opt.gpu_ids = [0]
    opt.serial_batches = True
    opt.dataset_mode = 'iemocap_miss'
    setattr(opt, 'miss_num', 'mix')
    modality = 'L'
    for cv in range(1, 11):
        opt.cvNo = cv
        teacher_path_cv = os.path.join(teacher_path, str(cv))
        dataset, val_dataset, tst_dataset = create_trn_val_tst_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        # model = MultiFusionMultiModel(opt)
        # model = NewTranslationModel(opt)
        model = EarlyFusionMultiModel(opt)
        model.cuda()
        model.load_networks_cv(teacher_path_cv)
        # extractor = MultiLayerFeatureExtractor(model, 'netC.module[4]')
        # save_root = 'analysis/teacher_feats/{}/'.format(modality) + str(cv)
        # if not os.path.exists(save_root):
        #     os.makedirs(save_root)

        # extract(model, dataset, save_root, phase='trn', modality=modality)
        # extract(model, val_dataset, save_root, phase='val', modality=modality)
        # extract(model, tst_dataset, save_root, phase='tst', modality=modality)

        save_root = '/data2/lrc/Iemocap_feature/early_fusion_reps_mix'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        
        # extract_all(model, dataset, save_root, cv, phase='trn')
        # extract_all_mix(model, val_dataset, save_root, cv, phase='val')
        # extract_all_mix(model, tst_dataset, save_root, cv, phase='tst')
        extract_all(model, dataset, save_root, cv, phase='trn')
        extract_all(model, val_dataset, save_root, cv, phase='val')
        extract_all(model, tst_dataset, save_root, cv, phase='tst')
        # extract_feat(model, dataset, 'latent', save_root, cv, phase='trn')
        # extract_feat(model, val_dataset, 'latent', save_root, cv, phase='val')
        # extract_feat(model, tst_dataset, 'latent', save_root, cv, phase='tst')

    