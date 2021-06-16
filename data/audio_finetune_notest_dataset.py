import torch
import numpy as np
from torch.utils.data import Dataset
from data.base_dataset import BaseDataset
from torch.nn.utils.rnn import pad_sequence

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

class AudioFinetuneNoTestDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--A_db_dir', type=str, help='where to load A_ft db')
        parser.add_argument('--data_type', type=str, choices=['iemocap', 'meld'], default='iemocap')
        return parser
    
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        
        self.comparE_env = lmdb.open(opt.A_db_dir,\
             readonly=True, create=False, readahead=False)
        self.comparE_txn = self.comparE_env.begin()
        # mask for text feature
        if opt.data_type == 'iemocap': 
            cvNo = opt.cvNo
            label_path = "/data4/lrc/IEMOCAP_features_npy/target/{}/"
            if set_name == 'val':
                self.label = np.load(label_path.format(cvNo) + "tst_label.npy")
                self.label = np.argmax(self.label, axis=1)
                self.int2name = np.load(label_path.format(cvNo) + "tst_int2name.npy")
                self.int2name = [x[0].decode() for x in self.int2name]
            else:
                label_trn = np.load(label_path.format(cvNo) + "trn_label.npy")
                label_val = np.load(label_path.format(cvNo) + "val_label.npy")
                self.label = np.concatenate([label_trn, label_val], axis=0)
                self.label = np.argmax(self.label, axis=1)
                int2name_trn = np.load(label_path.format(cvNo) + "trn_int2name.npy")
                int2name_val = np.load(label_path.format(cvNo) + "val_int2name.npy")
                self.int2name = np.concatenate([int2name_trn, int2name_val], axis=0)
                self.int2name = [x[0].decode() for x in self.int2name]
        else:
            label_path = "/data4/lrc/movie_dataset/downstream/MELD/target/{}/{}"
            name_map = {
                'trn': 'train',
                'val': "dev",
                'tst': 'test'
            }
            self.label = np.load(label_path.format(name_map[set_name], "label.npy"))
            self.int2name = np.load(label_path.format(name_map[set_name], "int2name.npy"))
            self.int2name = [name_map[set_name] + "_" + "dia"+x[0].split('_')[0] + '_' + "utt"+x[0].split('_')[1] for x in self.int2name]
        
        self.manual_collate_fn = True
        print(f"Finetune IEMOCAP dataset {set_name} created with total length: {len(self)}")
    
    def __getitem__(self, index):
        int2name = self.int2name[index]
        comparE_dump = msgpack.loads(self.comparE_txn.get(int2name.encode('utf8')), raw=False)
        comparE_ft = comparE_dump['comparE'].copy()
        acoustic = torch.from_numpy(comparE_ft).float()
        label = torch.tensor(self.label[index])
        index = torch.tensor(index)
        return {
            'A_feat': acoustic,
            'label': label,       
            'index': index,
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
        ft_type = ''
    
    opt = test()
    a = AudioFinetuneDataset(opt, 'trn')
    # import random
    # k = random.sample(range(len(a)), 1)[0] 
    # data = a[k]
    # int2name = a.int2name[k][0].decode()
    # print(int2name)
    # for k, v in data.items():
    #     if k not in ['int2name']:
    #         print(k, v.shape)
    #     else:
    #         print(k, v)

    batch = [a[0], a[1], a[2]]
    # import lmdb
    # import msgpack
    # import msgpack_numpy
    # msgpack_numpy.patch()
    # comparE_env = lmdb.open("/data4/lrc/movie_dataset/dbs/v2_0.5/comparE.db", readonly=True, create=False, readahead=False)
    # comparE_txn = comparE_env.begin(buffers=True)
    # dump = comparE_txn.get(int2name.encode('utf8'))
    # dump = msgpack.loads(dump, raw=False)
    # comparE = dump['comparE']
    # dataset_ft = data['A_feat']
    # comparE = torch.from_numpy(comparE.copy()).float()
    # print((dataset_ft==comparE).all())
    
    '''
    1: [1,2,3,4]
    2: [1,2,3],
    3: [4,5,6,7,8,0]
    4, [9,9]

    [[1,2,3,4,0,0],
     [1,2,3,0,0,0]
     ...
    '''