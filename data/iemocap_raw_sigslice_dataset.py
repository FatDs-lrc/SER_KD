import os.path as osp
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
from data.base_dataset import BaseDataset
from torch.nn.utils.rnn import pad_sequence

class IemocapRawSigSliceDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        return parser
    
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        label_path = "/data7/lrc/IEMOCAP_features_npy/target/{}/"
        # window and shift
        self.wlen = int(0.2 * 16000)     # 200ms window
        self.shift = int(0.01 * 16000)   # 10ms shift
        # mask for text feature
        self.data_root = '/data7/lrc/SincNet/IEMOCAP/norm_data'
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_label.npy")
        self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(label_path.format(cvNo) + f"{set_name}_int2name.npy")
        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")
        self.manual_collate_fn = True
    
    def __getitem__(self, index):
        utt_id = self.int2name[index][0].decode()
        wav_path = osp.join(self.data_root, self.get_wav_path(utt_id))
        segs = self.get_slices(wav_path, self.wlen, self.shift)
        label = torch.tensor(self.label[index])
        int2name = self.int2name[index][0].decode()
        return {
            'segments': torch.from_numpy(segs).float(),
            'label': torch.tensor(label),
            'int2name': int2name
        }
    
    def get_slices(self, wav_path, wlen, shift):
        signal, fs = sf.read(wav_path)
        assert fs == 16000, wav_path
        assert len(signal) - wlen >= 0, wav_path
        segs = []
        for st in range(0, int(len(signal)-wlen), int(shift)):
            seg = signal[st:st+wlen]
            segs.append(seg)
        return np.array(segs)
    
    def __len__(self):
        return len(self.label)
    
    def get_wav_path(self, utt_id):
        ses_id = utt_id[4]
        dialog_id = '_'.join(utt_id.split('_')[:-1])
        return osp.join(f'Session{ses_id}', 'sentences/wav', dialog_id, utt_id + '.wav')

    def collate_fn(self, batch):
        segs = [sample['segments'] for sample in batch]
        lengths = torch.tensor([len(sample) for sample in segs])
        mask = [torch.ones(x) for x in lengths]
        segs = pad_sequence(segs, batch_first=True, padding_value=0)
        mask = pad_sequence(mask, batch_first=True, padding_value=0)
        label = torch.tensor([sample['label'] for sample in batch])
        int2name = [sample['int2name'] for sample in batch]
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