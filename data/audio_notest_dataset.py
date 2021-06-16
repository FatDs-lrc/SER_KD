import torch
import numpy as np
from torch.utils.data import Dataset
from .base_dataset import BaseDataset

class AudioNoTestDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--ft_type', type=str, default='comparE_raw', help='which cross validation set')
        return parser
    
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val']
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        acoustic_ft_type = opt.ft_type

        data_path = "/data4/lrc/IEMOCAP_features_npy/feature/{}/{}/"
        label_path = "/data4/lrc/IEMOCAP_features_npy/target/{}/"
        if set_name == 'val':
            self.acoustic_data = np.load(data_path.format(acoustic_ft_type, cvNo) + f"tst.npy")
            self.label = np.load(label_path.format(cvNo) + f"tst_label.npy")
            self.label = np.argmax(self.label, axis=1)
            self.int2name = np.load(label_path.format(cvNo) + f"tst_int2name.npy")
        else:
            trn = np.load(data_path.format(acoustic_ft_type, cvNo) + "trn.npy")
            val = np.load(data_path.format(acoustic_ft_type, cvNo) + "val.npy")
            self.acoustic_data = np.concatenate([trn, val], axis=0)
            label_trn = np.load(label_path.format(cvNo) + "trn_label.npy")
            label_val = np.load(label_path.format(cvNo) + "val_label.npy")
            self.label = np.concatenate([label_trn, label_val], axis=0)
            self.label = np.argmax(self.label, axis=1)
            int2name_trn = np.load(label_path.format(cvNo) + "trn_int2name.npy")
            int2name_val = np.load(label_path.format(cvNo) + "val_int2name.npy")
            self.int2name = np.concatenate([int2name_trn, int2name_val], axis=0)
        # mask for text feature
        
        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")
    
    def __getitem__(self, index):
        acoustic = torch.from_numpy(self.acoustic_data[index]).float()
        label = torch.tensor(self.label[index])
        index = torch.tensor(index)
        int2name = self.int2name[index][0].decode()
        return {
            'A_feat': acoustic, 
            'label': label,
            'index': index,
            'int2name': int2name
        }
    
    def __len__(self):
        return len(self.label)

if __name__ == '__main__':
    class test:
        cvNo = 1
        ft_type = 'comparE_raw'
    
    opt = test()
    a = AudioNoTestDataset(opt, 'tst')
    data = a[0]
    for k, v in data.items():
        if k not in ['int2name']:
            print(k, v.shape)
        else:
            print(k, v)