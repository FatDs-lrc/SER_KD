import os
import os.path as osp
import shutil
import numpy as np
from tqdm import tqdm

def get_wav_path(utt_id):
    ses_id = utt_id[4]
    dialog_id = '_'.join(utt_id.split('_')[:-1])
    return osp.join(f'Session{ses_id}', 'sentences/wav', dialog_id, utt_id + '.wav')

def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(osp.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(osp.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    int2name = [x[0].decode() for x in int2name]
    assert len(int2name) == len(int2label)
    return int2name, int2label

def cp_wavs():
    data_root = '/data3/lrc/IEMOCAP_full_release/'
    target_root = '/data7/lrc/IEMOCAP_features_npy/target'
    save_root = '/data7/lrc/IEMOCAP_features_npy/wavs/raw'
    if not osp.exists(save_root):
        os.makedirs(save_root)
    trn_int2name, _ = get_trn_val_tst(target_root, 1, 'trn')
    val_int2name, _ = get_trn_val_tst(target_root, 1, 'val')
    tst_int2name, _ = get_trn_val_tst(target_root, 1, 'tst')
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    for utt_id in tqdm(all_utt_ids):
        src = osp.join(data_root, get_wav_path(utt_id))
        tgt = osp.join(save_root, utt_id + '.wav')
        shutil.copyfile(src, tgt)
    
cp_wavs()