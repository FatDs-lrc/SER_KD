import os
import os.path as osp
import shutil
import h5py
import numpy as np
import soundfile as sf
from tqdm import tqdm

def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(osp.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(osp.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    int2name = [x[0].decode() for x in int2name]
    assert len(int2name) == len(int2label)
    return int2name, int2label

def calc_mean_std():
    all_wavs = []
    wav_root = '/data6/lrc/IEMOCAP_features_npy/wavs/raw'
    target_root = '/data6/lrc/IEMOCAP_features_npy/target'
    trn_int2name, _ = get_trn_val_tst(target_root, 1, 'trn')
    val_int2name, _ = get_trn_val_tst(target_root, 1, 'val')
    tst_int2name, _ = get_trn_val_tst(target_root, 1, 'tst')
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    for utt_id in tqdm(all_utt_ids):
        src = osp.join(wav_root, utt_id + '.wav')
        sig, _ = sf.read(src)
        all_wavs.append(sig)
    all_wavs = np.concatenate(all_wavs)
    print(all_wavs.shape)
    mean = all_wavs.mean()
    std = all_wavs.std()
    print('MEAN:', mean)
    print('STD:', std)

calc_mean_std()