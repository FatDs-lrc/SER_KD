import torch
import numpy as np
from torch.utils.data import Dataset
from .base_dataset import BaseDataset

class AudioDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--ft_type', type=str, default='comparE_raw', help='which cross validation set')
        return parser
    
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        acoustic_ft_type = opt.ft_type
        # data_path = "/home/zzc/lrc/IEMOCAP_features_npy/feature/{}/{}/"
        # label_path = "/home/zzc/lrc/IEMOCAP_features_npy/target/{}/"

        data_path = "/data4/lrc/IEMOCAP_features_npy/feature/{}/{}/"
        label_path = "/data4/lrc/IEMOCAP_features_npy/target/{}/"
        self.acoustic_data = np.load(data_path.format(acoustic_ft_type, cvNo) + f"{set_name}.npy")

        # mask for text feature
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_label.npy")
        self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(label_path.format(cvNo) + f"{set_name}_int2name.npy")
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
        ft_type = 'comparE_downsampled'
    
    opt = test()
    a = AudioDataset(opt, 'trn')
    data = a[0]
    for k, v in data.items():
        if k not in ['int2name']:
            print(k, v.shape)
        else:
            print(k, v)