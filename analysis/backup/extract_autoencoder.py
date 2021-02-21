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
from models.simple_mapping_model import SimpleMappingModel
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_recon_latent(model, val_iter, save_path, cv, phase='val'):
    model.eval()
    recon_feats = []
    latent_feats = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        recon = getattr(model, 'recon').detach().cpu().numpy()
        latent = getattr(model, 'latent').detach().cpu().numpy()
        recon_feats.append(recon)
        latent_feats.append(latent)
        
    recon_feats = np.vstack(recon_feats)
    latent_feats = np.vstack(latent_feats)
    
    # save model_feat
    feat_save_root = os.path.join(save_path, str(cv))
    if not os.path.exists(feat_save_root):
        os.mkdir(feat_save_root)

    feat_save_path = os.path.join(feat_save_root, '{}_{{}}.npy'.format(phase))
    np.save(feat_save_path.format('recon'), recon_feats)
    np.save(feat_save_path.format('latent'), latent_feats)

def load_from_opt_record(file_path):
    opt_content = json.load(open(file_path, 'r'))
    opt = OptConfig()
    opt.load(opt_content)
    return opt


if __name__ == '__main__':
    # teacher_path = 'checkpoints/ef_AVL_Adnn512,256,128_Vlstm128_maxpool_Lcnn128_fusion256,128run1/'
    model_type = 'A2L'
    # teacher_path = 'checkpoints/simple_map_{}_run1'.format(model_type)
    teacher_path = 'checkpoints/simple_map_new_A2L_mse0.1_cycle0.1_run1'
    opt_path = os.path.join(teacher_path, 'train_opt.conf')
    opt = load_from_opt_record(opt_path)
    opt.isTrain = False                             # teacher model should be in test mode
    opt.gpu_ids = [0]
    opt.serial_batches = True
    for cv in range(1, 11):
        opt.cvNo = cv
        teacher_path_cv = os.path.join(teacher_path, str(cv))
        dataset, val_dataset, tst_dataset = create_trn_val_tst_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = SimpleMappingModel(opt)
        model.cuda()
        model.load_networks_cv(teacher_path_cv)
        
        save_root = '/data2/lrc/Iemocap_feature/simple_map_analysis/{}'.format(model_type)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        
        extract_recon_latent(model, dataset, save_root, cv, phase='trn')
        extract_recon_latent(model, val_dataset, save_root, cv, phase='val')
        extract_recon_latent(model, tst_dataset, save_root, cv, phase='tst')
    