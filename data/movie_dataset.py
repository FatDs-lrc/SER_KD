import os.path as osp
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from data.base_dataset import BaseDataset
from torch.nn.utils.rnn import pad_sequence

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

class MovieDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--A_db_dir', type=str, help='which cross validation set')
        parser.add_argument('--L_db_dir', type=str, help='which cross validation set')
        return parser

    def __init__(self, opt, set_name):
        ''' movie_dataset reader set_name in ['trn', 'val']
        '''
        super().__init__(opt)
        if set_name == 'trn':
            A_db_dir = opt.A_db_dir
            L_db_dir = opt.L_db_dir
        else:
            A_db_dir = "/data4/lrc/movie_dataset/dbs/v2_2/comparE.db"
            L_db_dir = "/data4/lrc/movie_dataset/dbs/v2_2/bert_light.db"

        self.comparE_env = lmdb.open(A_db_dir,\
             readonly=True, create=False, readahead=False)
        self.roberta_env = lmdb.open(L_db_dir,\
             readonly=True, create=False, readahead=False)
        self.comparE_txn = self.comparE_env.begin()
        self.roberta_txn = self.roberta_env.begin()
        self.int2name = json.load(open(osp.join(L_db_dir, 'id.json')))
        self.manual_collate_fn = True
        print(f"EmoMovie dataset created with total length: {len(self)}")
    
    def __getitem__(self, index):
        name = self.int2name[index]
        comparE_dump = msgpack.loads(self.comparE_txn.get(name.encode('utf8')), raw=False)
        roberta_dump = msgpack.loads(self.roberta_txn.get(name.encode('utf8')), raw=False)
        logits = roberta_dump['logits'].copy()
        input_ids = roberta_dump['input_ids'][:-1].copy()
        comparE = comparE_dump['comparE'].copy()
        return {
            'int2name': name,
            'logits': torch.from_numpy(logits).float(),
            'input_ids': torch.from_numpy(input_ids).long(),
            'comparE': torch.from_numpy(comparE).float(),
            'label': torch.argmax(torch.from_numpy(logits)),
        }

    def __len__(self):
        return len(self.int2name)
    
    def collate_fn(self, batch):
        int2name = [sample['int2name'] for sample in batch]
        comparE = [sample['comparE'] for sample in batch]
        comparE = pad_sequence(comparE, batch_first=True, padding_value=0)
        len_comparE = torch.tensor([len(sample['comparE']) for sample in batch]).long()
        logits = [sample['logits'] for sample in batch]
        logits = torch.cat(logits)
        label = torch.tensor([sample['label'] for sample in batch]).long()
        input_ids = [sample['input_ids'] for sample in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        input_ids = torch.cat([input_ids, torch.ones(input_ids.size(0), 1).long()*102], dim=1)
        mask = (input_ids != 0).long()
        return {
            'int2name': int2name,
            'logits': logits,
            'comparE': comparE,
            'len_comparE': len_comparE,
            'label': label,
            'input_ids': input_ids,
            'mask': mask
        }

if __name__ == '__main__':
    class test:
        A_db_dir = '/data4/lrc/movie_dataset/dbs/v2_2/comparE.db'
        L_db_dir = '/data4/lrc/movie_dataset/dbs/v2_2/bert_light.db'
    
    opt = test()
    a = MovieDataset(opt)
    data0 = a[0]
    data1 = a[1]
    data2 = a[2]
    print('-----------------data0-----------------------')
    for k, v in data0.items():
        if k not in ['int2name', 'input_ids']:
            print(k, v.shape)
        else:
            print(k, v)
    
    print('-----------------data1-----------------------')
    for k, v in data1.items():
        if k not in ['int2name', 'input_ids']:
            print(k, v.shape)
        else:
            print(k, v)
    print('-----------------data2-----------------------')
    for k, v in data2.items():
        if k not in ['int2name', 'input_ids']:
            print(k, v.shape)
        else:
            print(k, v)
    print('---------------------------------------------')
    collatn = a.collate_fn([data0, data1, data2])
    print('-----------------collatn-----------------------')
    for k, v in collatn.items():
        if k not in ['int2name', 'input_ids', 'mask']:
            print(k, v.shape)
        else:
            print(k, v)
    print('---------------------------------------------')