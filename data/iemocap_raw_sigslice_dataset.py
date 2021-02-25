import os.path as osp
import torch
import numpy as np
import soundfile as sf
import sys
sys.path.append('/data6/lrc/SER_KD')
from torch.utils.data import Dataset
from data.base_dataset import BaseDataset
from torch.nn.utils.rnn import pad_sequence

class IemocapRawSigSliceDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--norm_sig', type=str, default='None', choices=['None', 'utt', 'all'], help='which cross validation set')
        return parser
    
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        self.norm = opt.norm_sig
        label_path = "/data6/lrc/IEMOCAP_features_npy/target/{}/"
        # window and shift
        self.wlen = int(0.2 * 16000)     # 200ms window
        self.shift = int(0.1 * 16000)   # 100ms shift
        # mask for text feature
        self.data_root = '/data6/lrc/IEMOCAP_features_npy/wavs/raw/'
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_label.npy")
        self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(label_path.format(cvNo) + f"{set_name}_int2name.npy")
        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")
        self.manual_collate_fn = True
    
    def __getitem__(self, index):
        utt_id = self.int2name[index][0].decode()
        wav_path = osp.join(self.data_root, utt_id + '.wav')
        segs = torch.from_numpy(self.get_slices(wav_path, self.wlen, self.shift)).float()
        label = torch.tensor(self.label[index])
        int2name = self.int2name[index][0].decode()
        return {
            'segments': segs,
            'label': label,
            'int2name': int2name
        }
    
    def get_slices(self, wav_path, wlen, shift):
        signal, fs = sf.read(wav_path)
        assert fs == 16000, wav_path
        if self.norm == 'None':
            pass
        elif self.norm == 'utt':
            mean = signal.mean()
            std = signal.std()
            signal = (signal - mean) / std
        elif self.norm == 'all':
            mean = -7.395432666646425e-06
            std = 0.058110240439255334
            signal = (signal - mean) / std
       
        if len(signal) < wlen:
            segs = np.concatenate([signal, np.zeros(wlen-len(signal))], axis=0)
            segs = np.expand_dims(segs, axis=0)
        else:
            segs = []
            for st in range(0, int(len(signal)-wlen), int(shift)):
                seg = signal[st:st+wlen]
                segs.append(seg)
        return np.array(segs)
    
    def __len__(self):
        return len(self.label)

    def collate_fn(self, batch):
        max_len = 40
        segs = [sample['segments'] for sample in batch]
        lengths = torch.tensor([len(sample) for sample in segs])
        mask = [torch.ones(x) for x in lengths]
        segs = pad_sequence(segs, batch_first=True, padding_value=0)
        mask = pad_sequence(mask, batch_first=True, padding_value=0)
        label = torch.tensor([sample['label'] for sample in batch])
        int2name = [sample['int2name'] for sample in batch]
        if segs.size(1) > max_len:
            segs = segs[:, :max_len, :]
        # print(segs.shape)
        return {
            'segments': segs,
            'lengths': lengths,
            'mask': mask,
            'label': label,
            'int2name': int2name
        }

if __name__ == '__main__':
    class test:
        cvNo = 1
        norm_sig='all'
    opt = test()
    a = IemocapRawSigSliceDataset(opt, 'trn')
    # data = a[0]
    # for k, v in data.items():
    #     if k not in ['int2name']:
    #         print(k, v.shape)
    #     else:
    #         print(k, v)
    batch = [a[x] for x in range(5)]
    batch_data = a.collate_fn(batch)
    for k, v in batch_data.items():
        if k not in ['int2name']:
            print(k, v.shape)
        else:
            print(k, v)
    print(batch_data['mask'][0])