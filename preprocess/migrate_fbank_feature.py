import os
import os.path as osp
import numpy as np


def statistic():
    src_root = '/data7/ser/exp/las/features/fbank_cmvnTrue/1'
    trn = np.load(osp.join(src_root, 'trn_ft.npy'), allow_pickle=True)
    val = np.load(osp.join(src_root, 'val_ft.npy'), allow_pickle=True)
    tst = np.load(osp.join(src_root, 'test_ft.npy'), allow_pickle=True)
    trn = [x.shape[0] for x in trn]
    val = [x.shape[0] for x in val]
    tst = [x.shape[0] for x in tst]
    all_lengths = trn + val + tst
    all_lengths = sorted(all_lengths)
    print('MIN:', min(all_lengths))
    print('MAX:', max(all_lengths))
    print('AVG:', sum(all_lengths)/len(all_lengths))
    print('75%:', all_lengths[int(len(all_lengths)*0.75)])
    print('90%:', all_lengths[int(len(all_lengths)*0.9)])

def padding_to_fixlen(data, length):
    if len(data) >= length:
        ret = data[:length]
    else:
        ret = np.concatenate([data, np.zeros([length-len(data), data.shape[1]])], axis=0)
    return ret

def migrate():
    max_len = 600
    src_root = '/data7/ser/exp/las/features/fbank_cmvnTrue'
    tgt_root = '/data7/lrc/IEMOCAP_features_npy/feature/fbank_raw'
    for cv in range(1, 11):
        src_dir = osp.join(src_root, str(cv))
        tgt_dir = osp.join(tgt_root, str(cv))
        if not osp.exists(tgt_dir):
            os.makedirs(tgt_dir)
        for set_name in ['trn', 'val', 'test']:
            src_file = osp.join(src_dir, f'{set_name}_ft.npy')
            src_data = np.load(src_file, allow_pickle=True)
            tgt_data = np.array([padding_to_fixlen(x, max_len) for x in src_data])
            save_path = osp.join(tgt_dir, f'{set_name}.npy') \
                if set_name != 'test' else osp.join(tgt_dir, 'tst.npy')
            np.save(save_path, tgt_data)
            print('Save to {} with shape {}'.format(save_path, tgt_data.shape))

def norm():
    src_root = '/data7/lrc/IEMOCAP_features_npy/feature/fbank_raw'
    tgt_root = '/data7/lrc/IEMOCAP_features_npy/feature/fbank'
    for cv in range(1, 11):
        src_dir = osp.join(src_root, str(cv))
        tgt_dir = osp.join(tgt_root, str(cv))
        if not osp.exists(tgt_dir):
            os.makedirs(tgt_dir)
        
        trn = np.load(osp.join(src_dir, 'trn.npy'))
        val = np.load(osp.join(src_dir, 'val.npy'))
        tst = np.load(osp.join(src_dir, 'tst.npy'))
        mean = np.mean(trn.reshape([-1, trn.shape[-1]]), axis=0)
        std = np.std(trn.reshape([-1, trn.shape[-1]]), axis=0)
        print(mean.shape, std.shape)
        std[std==0.0] = 1.0
        trn = (trn - mean) / std
        val = (val - mean) / std
        tst = (tst - mean) / std
        np.save(osp.join(tgt_dir, 'trn.npy'), trn)
        np.save(osp.join(tgt_dir, 'val.npy'), val)
        np.save(osp.join(tgt_dir, 'tst.npy'), tst)
        print('Save to {} with shape {}'.format(osp.join(tgt_dir, 'trn.npy'), trn.shape))
        print('Save to {} with shape {}'.format(osp.join(tgt_dir, 'val.npy'), val.shape))
        print('Save to {} with shape {}'.format(osp.join(tgt_dir, 'tst.npy'), tst.shape))


# statistic()
'''
MIN: 56
MAX: 3412
AVG: 452.89784849032725
75%: 575
90%: 867
'''
# migrate()
norm()
