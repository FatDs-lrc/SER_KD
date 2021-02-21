import os
import os.path as osp
import shutil
import h5py
import numpy as np
import soundfile as sf
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
    target_root = '/data6/lrc/IEMOCAP_features_npy/target'
    save_root = '/data6/lrc/IEMOCAP_features_npy/wavs/raw'
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

def save_as_np():
    wav_root = '/data6/lrc/IEMOCAP_features_npy/wavs/raw'
    save_root = '/data6/lrc/IEMOCAP_features_npy/wavs/npy'
    target_root = '/data6/lrc/IEMOCAP_features_npy/target'
    if not osp.exists(save_root):
        os.makedirs(save_root)
    trn_int2name, _ = get_trn_val_tst(target_root, 1, 'trn')
    val_int2name, _ = get_trn_val_tst(target_root, 1, 'val')
    tst_int2name, _ = get_trn_val_tst(target_root, 1, 'tst')
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    for utt_id in tqdm(all_utt_ids):
        src = osp.join(wav_root, utt_id + '.wav')
        tgt = osp.join(save_root, utt_id + '.npy')
        sig, _ = sf.read(src)
        np.save(tgt, sig)

def save_as_h5():
    wav_root = '/data6/lrc/IEMOCAP_features_npy/wavs/raw'
    save_root = '/data6/lrc/IEMOCAP_features_npy/wavs/h5'
    target_root = '/data6/lrc/IEMOCAP_features_npy/target'
    if not osp.exists(save_root):
        os.makedirs(save_root)
    trn_int2name, _ = get_trn_val_tst(target_root, 1, 'trn')
    val_int2name, _ = get_trn_val_tst(target_root, 1, 'val')
    tst_int2name, _ = get_trn_val_tst(target_root, 1, 'tst')
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    h5f = h5py.File(osp.join(save_root, 'all.h5'), 'w')
    for utt_id in tqdm(all_utt_ids):
        src = osp.join(wav_root, utt_id + '.wav')
        sig, _ = sf.read(src)
        h5f[utt_id] = sig
    h5f.close()

def test_reading_time():
    import time
    wav_root = '/data6/lrc/IEMOCAP_features_npy/wavs/raw'
    npy_root = '/data6/lrc/IEMOCAP_features_npy/wavs/npy'
    h5_root = '/data6/lrc/IEMOCAP_features_npy/wavs/h5'
    target_root = '/data6/lrc/IEMOCAP_features_npy/target'
    trn_int2name, _ = get_trn_val_tst(target_root, 1, 'trn')
    val_int2name, _ = get_trn_val_tst(target_root, 1, 'val')
    tst_int2name, _ = get_trn_val_tst(target_root, 1, 'tst')
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    all_wavs = [osp.join(wav_root, utt_id + '.wav') for utt_id in all_utt_ids]
    all_npys = [osp.join(npy_root, utt_id + '.npy') for utt_id in all_utt_ids]
    h5f = h5py.File(osp.join(h5_root, 'all.h5'), 'r')
    print('READING WAVS:')
    start = time.time()
    for wav in all_wavs:
        sig, _ = sf.read(wav)
    end = time.time()
    print(end-start)
    print('-------------------')
    print('READING NPYS:')
    start = time.time()
    for npy in all_npys:
        y = np.load(npy)
    end = time.time()
    print(end-start)
    print('-------------------')
    print('READING H5:')
    start = time.time()
    for utt_id in all_utt_ids:
        y = h5f[utt_id][()]
    end = time.time()
    print(end-start)
    
    


if __name__ == '__main__':
    # cp_wavs()
    # save_as_np()
    # save_as_h5()
    test_reading_time()