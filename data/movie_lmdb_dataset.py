import os.path as osp
import torch
import h5py
import numpy as np
import json
from torch.utils.data import Dataset
from data.base_dataset import BaseDataset
from torch.nn.utils.rnn import pad_sequence

class MovieLMDBDataset(BaseDataset):

    def __init__(self, opt, set_name):
        ''' movie_dataset reader set_name in ['trn', 'val']
        '''
        super().__init__(opt)
        data_root = "/data7/lrc/movie_dataset/filtered_data"
        name_dir = osp.join(data_root, 'name')
        self.hidden_state_layers = [5,8,11,13]
        self.roberta_ft_dir = osp.join(data_root, 'roberta')
        self.comparE_ft_dir = osp.join(data_root, 'comparE')
        self.int2name = json.load(open(osp.join(name_dir, set_name + '.json')))
        self.movie_names = set([x.split('+')[0] for x in self.int2name])
        self.manual_collate_fn = True
        print(f"EmoMovie dataset {set_name} created with total length: {len(self)}")
    
    def __getitem__(self, index):
        name = self.int2name[index]
        roberta_file = osp.join(self.roberta_ft_dir, name.replace('+', '/') + '.npz')
        comparE_file = osp.join(self.comparE_ft_dir, name.replace('+', '/') + '.npz')
        roberta_data = np.load(roberta_file)
        comparE_data = np.load(comparE_file)
        logits = roberta_data['logits']
        hidden_states = roberta_data['hidden_states']
        comparE = comparE_data['comparE']
        return {
            'int2name': name,
            'logits': torch.from_numpy(logits).float(),
            'hidden_states': torch.from_numpy(hidden_states).float(),
            'comparE': torch.from_numpy(comparE).float(),
            'label': torch.argmax(torch.from_numpy(logits))
        }

    def __len__(self):
        return len(self.int2name)
    
    def collate_fn(self, batch):
        int2name = [sample['int2name'] for sample in batch]
        logits = [sample['logits'] for sample in batch]
        hidden_states = [sample['hidden_states'] for sample in batch]
        comparE = [sample['comparE'] for sample in batch]
        logits = torch.cat(logits)
        label = torch.tensor([sample['label'] for sample in batch]).long()
        len_hidden_states = torch.tensor([len(sample['hidden_states'][0]) for sample in batch]).long()
        len_comparE = torch.tensor([len(sample['comparE']) for sample in batch]).long()
        all_layer = []
        for layer in range(self.hidden_state_layers):
            layer_hidden = [x[layer] for x in hidden_states]
            layer_hidden = pad_sequence(layer_hidden, batch_first=True, padding_value=0)
            all_layer.append(layer_hidden.unsqueeze(1))

        hidden_states = torch.cat(all_layer, dim=1)
        comparE = pad_sequence(comparE, batch_first=True, padding_value=0)
    
        # print(segs.shape)
        return {
            'int2name': int2name,
            'logits': logits,
            # 'hidden_states': hidden_states,
            'comparE': comparE,
            'len_hidden_states': len_hidden_states,
            'len_comparE': len_comparE,
            'label': label
        }

if __name__ == '__main__':
    class test:
        cvNo = 1
        ft_type = 'comparE_downsampled'
    
    opt = test()
    a = MovieDataset(opt, 'trn')
    # data0 = a[0]
    # data1 = a[1]
    # data2 = a[2]
    # print('-----------------data0-----------------------')
    # for k, v in data0.items():
    #     if k not in ['int2name', 'len_hidden_states', 'len_hidden_states']:
    #         print(k, v.shape)
    #     else:
    #         print(k, v)
    # print('-----------------data1-----------------------')
    # for k, v in data1.items():
    #     if k not in ['int2name', 'len_hidden_states', 'len_hidden_states']:
    #         print(k, v.shape)
    #     else:
    #         print(k, v)
    # print('-----------------data2-----------------------')
    # for k, v in data2.items():
    #     if k not in ['int2name', 'len_hidden_states', 'len_hidden_states']:
    #         print(k, v.shape)
    #     else:
    #         print(k, v)
    # print('---------------------------------------------')
    # collatn = a.collate_fn([data0, data1, data2])
    # print('-----------------collatn-----------------------')
    # for k, v in collatn.items():
    #     if k not in ['int2name', 'len_hidden_states', 'len_comparE', 'label', 'logits']:
    #         print(k, v.shape)
    #     else:
    #         print(k, v)
    # print('---------------------------------------------')
    from tqdm import tqdm
    for data in tqdm(a):
        pass