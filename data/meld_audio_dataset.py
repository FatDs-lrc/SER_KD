import torch
import numpy as np
from torch.utils.data import Dataset
from data.base_dataset import BaseDataset
from torch.nn.utils.rnn import pad_sequence

class MeldAudioDataset(BaseDataset):
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
        self.data_path = "/data4/lrc/MELD/feature/comparE_norm/{}"
        label_path = "/data4/lrc/MELD/target/{}/{}"
        name_map = {
            'trn': 'train',
            'val': "dev",
            'tst': 'test'
        }
        # mask for text feature
        self.label = np.load(label_path.format(name_map[set_name], "label.npy"))
        self.int2name = np.load(label_path.format(name_map[set_name], "int2name.npy"))
        self.int2name = [name_map[set_name] + "_" + "dia"+x[0].split('_')[0] + '_' + "utt"+x[0].split('_')[1] for x in self.int2name]
        self.manual_collate_fn = True
        print(f"MELD Audio dataset {set_name} created with total length: {len(self)}")
    
    def __getitem__(self, index):
        int2name = self.int2name[index]
        ft = np.load(self.data_path.format(int2name + '.npy'))
        acoustic = torch.from_numpy(ft).float()
        label = torch.tensor(self.label[index])
        return {
            'A_feat': acoustic, 
            'label': label,
            'int2name': int2name
        }
    
    def __len__(self):
        return len(self.label)
    
    def collate_fn(self, batch):
        int2name = [sample['int2name'] for sample in batch]
        A_feat = [sample['A_feat'] for sample in batch]
        A_feat = pad_sequence(A_feat, batch_first=True, padding_value=0)
        label = torch.tensor([sample['label'] for sample in batch]).long()

        return {
            'A_feat': A_feat, 
            'label': label,
            'int2name': int2name
        }

if __name__ == '__main__':
    class test:
        cvNo = 1
        ft_type = 'comparE_downsampled'
    
    opt = test()
    a = MeldAudioDataset(opt, 'trn')
    data = a[0]
    for k, v in data.items():
        if k not in ['int2name']:
            print(k, v.shape)
        else:
            print(k, v)
    data1 = a[1]
    data2 = a[2]
    collate = a.collate_fn([a[0], a[1], a[2]])

    for k, v in collate.items():
        if k not in ['int2name']:
            print(k, v.shape)
        else:
            print(k, v)