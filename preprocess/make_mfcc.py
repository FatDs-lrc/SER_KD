import os
import h5py
import json
import numpy as np
import pandas as pd
import scipy.signal as spsig
from tqdm import tqdm
import torchaudio
torchaudio.set_audio_backend("sox")
from audio_tool import Delta

class MfccExtractor(object):  
    def __init__(self, num_bins=20, sample_frequency=16000, window=0.025, shift=0.01, delta=2):
        self.num_bins = num_bins
        self.sample_frequency = sample_frequency
        self.window = window * 1000
        self.shift = shift * 1000
        if delta > 0:
            self.delta = Delta(delta)
    
    def __call__(self, wav_file):
        waveform, sample_rate = torchaudio.load(wav_file)
        assert sample_rate == self.sample_frequency, 'sample_rate different from {}'.format(self.sample_frequency)
        y = torchaudio.compliance.kaldi.mfcc(
            waveform,
            num_ceps=self.num_bins,
            channel=-1,
            sample_frequency=sample_rate,
            frame_length=self.window,
            frame_shift=self.shift
        )
        delta = self.delta(y.unsqueeze(0)).permute(1, 0, 2)
        # print(delta.size())
        delta = delta.contiguous().view(delta.size(0), -1)
        return delta.numpy()


def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label

def make_all_mfcc(config):
    extractor = MfccExtractor()
    trn_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'trn')
    val_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'tst')
    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
    tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    save_dir = os.path.join(config['feature_root'], 'feature', 'mfcc')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_h5f = h5py.File(os.path.join(save_dir, 'all.h5'), 'w')
    for utt_id in tqdm(all_utt_ids):
        ses_id = utt_id[4]
        dialog_id = '_'.join(utt_id.split('_')[:-1])
        wav_path = os.path.join(config['data_root'], f'Session{ses_id}', 'sentences', 'wav', f'{dialog_id}', f'{utt_id}.wav')
        feat = extractor(wav_path)
        all_h5f[utt_id] = feat

def normlize_on_trn(config, input_file, output_file):
    h5f = h5py.File(output_file, 'w')
    in_data = h5py.File(input_file, 'r')
    for cv in range(1, 11):
        print(cv)
        trn_int2name, _ = get_trn_val_tst(config['target_root'], cv, 'trn')
        trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
        all_feat = [in_data[utt_id][()] for utt_id in trn_int2name]
        all_feat = np.concatenate(all_feat, axis=0)
        mean_f = np.mean(all_feat, axis=0)
        std_f = np.std(all_feat, axis=0)
        std_f[std_f == 0.0] = 1.0
        cv_group = h5f.create_group(str(cv))
        cv_group['mean'] = mean_f
        cv_group['std'] = std_f
        print("mean:", mean_f)
        print("std:", std_f)

def padding_to_fixlen(feat, max_len):
    assert feat.ndim == 2
    if feat.shape[0] >= max_len:
        feat = feat[:max_len]
    else:
        feat = np.concatenate([feat, \
            np.zeros((max_len-feat.shape[0], feat.shape[1]))], axis=0)
    return feat

def migrate_mfcc_to_npy(config):
    max_len = 600
    feat_path = os.path.join(config['feature_root'], 'feature', 'mfcc', 'all.h5')
    mean_std_path = os.path.join(config['feature_root'], 'feature', 'mfcc', 'mean_std.h5')
    feat_h5f = h5py.File(feat_path, 'r')
    mean_std = h5py.File(mean_std_path, 'r')
    for cv in range(1, 11):
        save_dir = os.path.join(config['feature_root'], 'feature', 'mfcc', str(cv)) # f'/data3/lrc/Iemocap_feature/cv_level/feature/mfcc/{cv}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        mean = mean_std[str(cv)]['mean'][()]
        std = mean_std[str(cv)]['std'][()]
        for part in ['trn', 'val', 'tst']:
            part_feat = []
            int2name, _ = get_trn_val_tst(config['target_root'], cv, part)
            int2name = [x[0].decode() for x in int2name]
            for utt_id in tqdm(int2name):
                feat = feat_h5f[utt_id][()]
                feat = (feat-mean)/std
                feat = padding_to_fixlen(feat, max_len)
                part_feat.append(feat)
            part_feat = np.array(part_feat)
            print(f"cv: {cv} {part} {part_feat.shape}")
            save_path = os.path.join(save_dir, f"{part}.npy")
            np.save(save_path, part_feat)


def statis_mfcc(config):
    path = os.path.join(config['feature_root'], 'feature', 'mfcc', 'all.h5')
    h5f = h5py.File(path, 'r')
    lengths = []
    for utt_id in h5f.keys():
        lengths.append(h5f[utt_id][()].shape[0])
    lengths = sorted(lengths)
    print('MIN:', min(lengths))
    print('MAX:', max(lengths))
    print('MEAN: {:.2f}'.format(sum(lengths) / len(lengths)))
    print('50%:', lengths[len(lengths)//2])
    print('75%:', lengths[int(len(lengths)*0.75)])
    print('90%:', lengths[int(len(lengths)*0.9)])


if __name__ == '__main__':
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../', 'data/config', 'IEMOCAP_config.json')
    config = json.load(open(config_path))
    make_all_mfcc(config)
    statis_mfcc(config)
    normlize_on_trn(config, os.path.join(config['feature_root'], 'feature', 'mfcc', 'all.h5'), os.path.join(config['feature_root'], 'feature', 'mfcc', 'mean_std.h5'))
    migrate_mfcc_to_npy(config)
    # extractor = MfccExtractor()
    # wav_path = '/data6/lrc/IEMOCAP_features_npy/wavs/raw/Ses02M_impro08_M027.wav'
    # ft = extractor(wav_path)
    # print(ft.shape)