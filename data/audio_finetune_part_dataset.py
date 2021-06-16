import torch
import numpy as np
from torch.utils.data import Dataset
from data.base_dataset import BaseDataset
from torch.nn.utils.rnn import pad_sequence

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

class AudioFinetunePartDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--A_db_dir', type=str, help='where to load A_ft db')
        parser.add_argument('--data_type', type=str, choices=['iemocap', 'meld'], default='iemocap')
        parser.add_argument('--ratio', type=str, choices=['1-2', '1-3', '1-5'], default='1-2')
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
            label_path = "/data4/lrc/IEMOCAP_features_npy/target{}/{}/"
            # self.label = np.load(label_path.format(opt.ratio, cvNo) + f"{set_name}_label.npy")
            # self.int2name = np.load(label_path.format(opt.ratio, cvNo) + f"{set_name}_int2name.npy")
            if set_name == 'val':
                self.label = np.load(label_path.format(opt.ratio, cvNo) + "tst_label.npy")
                self.int2name = np.load(label_path.format(opt.ratio, cvNo) + "tst_int2name.npy")
            else:
                label_trn = np.load(label_path.format(opt.ratio, cvNo) + "trn_label.npy")
                label_val = np.load(label_path.format(opt.ratio, cvNo) + "val_label.npy")
                self.label = np.concatenate([label_trn, label_val], axis=0)
                int2name_trn = np.load(label_path.format(opt.ratio, cvNo) + "trn_int2name.npy")
                int2name_val = np.load(label_path.format(opt.ratio, cvNo) + "val_int2name.npy")
                self.int2name = np.concatenate([int2name_trn, int2name_val], axis=0)
        else:
            label_path = "/data4/lrc/movie_dataset/downstream/MELD/target{}/{}/{}"
            name_map = {
                'trn': 'train',
                'val': "dev",
                'tst': 'test'
            }
            self.label = np.load(label_path.format(opt.ratio, name_map[set_name], "label.npy"))
            self.int2name = np.load(label_path.format(opt.ratio, name_map[set_name], "int2name.npy"))
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
        A_db_dir = '/data4/lrc/movie_dataset/dbs/iemocap/comparE.db'
        data_type = 'iemocap'
        cvNo = 1
        ratio = '1-2'
    
    opt = test()
    a = AudioFinetunePartDataset(opt, 'tst')
    print(len(a))
    import random
    k = random.sample(range(len(a)), 1)[0] 
    data = a[k]
    int2name = a.int2name[k]
    print(int2name)
    for k, v in data.items():
        if k not in ['int2name']:
            print(k, v.shape)
        else:
            print(k, v)

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
    