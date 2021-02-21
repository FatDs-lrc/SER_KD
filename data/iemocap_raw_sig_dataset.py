import os.path as osp
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
from data.base_dataset import BaseDataset
from torch.nn.utils.rnn import pad_sequence

class IemocapRawSigDataset(BaseDataset):
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
        # mask for text feature
        self.data_root = '/data3/lrc/IEMOCAP_full_release/'
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_label.npy")
        self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(label_path.format(cvNo) + f"{set_name}_int2name.npy")
        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")
        self.manual_collate_fn = True
    
    def __getitem__(self, index):
        utt_id = self.int2name[index][0].decode()
        wav_path = osp.join(self.data_root, self.get_wav_path(utt_id))
        signal, _ = sf.read(wav_path)
        # signal = signal.astype(np.float64)
        # signal = signal/np.max(np.abs(signal))
        signal = torch.from_numpy(signal).float()
        label = torch.tensor(self.label[index])
        int2name = self.int2name[index][0].decode()
        return {
            'signal': signal,
            'label': label,
            'int2name': int2name
        }
    
    def __len__(self):
        return len(self.label)
    
    def get_wav_path(self, utt_id):
        ses_id = utt_id[4]
        dialog_id = '_'.join(utt_id.split('_')[:-1])
        return osp.join(f'Session{ses_id}', 'sentences/wav', dialog_id, utt_id + '.wav')

    def collate_fn(self, batch):
        sigs = [sample['signal'] for sample in batch]
        lengths = torch.tensor([len(sample) for sample in sigs])
        mask = [torch.ones(x) for x in lengths]
        sigs = pad_sequence(sigs, batch_first=True, padding_value=0)
        mask = pad_sequence(mask, batch_first=True, padding_value=0)
        label = torch.tensor([sample['label'] for sample in batch])
        int2name = [sample['int2name'] for sample in batch]
        return {
            'signal': sigs,
            'lengths': lengths,
            'mask': mask,
            'label': label,
            'int2name': int2name
        }

if __name__ == '__main__':
    class test:
        cvNo = 1
    
    opt = test()
    a = IemocapRawSigDataset(opt, 'trn')
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
    print(batch_data['signal'][0])
    print(batch_data)