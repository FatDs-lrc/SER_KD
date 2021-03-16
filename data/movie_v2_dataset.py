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

class MovieV2Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--data_type', type=str, default='name', help='which cross validation set')
        return parser

    def __init__(self, opt, set_name):
        ''' movie_dataset reader set_name in ['trn', 'val']
        '''
        super().__init__(opt)
        
        data_root = "/data4/lrc/movie_dataset"
        name_dir = osp.join(data_root, 'filtered_data', opt.data_type)
        self.hidden_state_layers = [6, 8, 10, 12]
        self.comparE_env = lmdb.open(osp.join(data_root, "dbs/v2/comparE.db"),\
             readonly=True, create=False, readahead=False)
        self.roberta_env = lmdb.open(osp.join(data_root, "dbs/v2/bert_light.db"),\
             readonly=True, create=False, readahead=False)
        self.comparE_txn = self.comparE_env.begin()
        self.roberta_txn = self.roberta_env.begin()
        self.int2name = json.load(open(osp.join(name_dir, set_name + '.json')))
        self.manual_collate_fn = True
        print(f"EmoMovie dataset {set_name} created with total length: {len(self)}")
    
    def __getitem__(self, index):
        name = self.int2name[index]
        comparE_dump = msgpack.loads(self.comparE_txn.get(name.encode('utf8')), raw=False)
        roberta_dump = msgpack.loads(self.roberta_txn.get(name.encode('utf8')), raw=False)
        logits = roberta_dump['logits'].copy()
        input_ids = roberta_dump['input_ids'][:-1].copy()
        # hidden_states = roberta_dump['hidden_states'].copy()[self.hidden_state_layers]
        comparE = comparE_dump['comparE'].copy()
        return {
            'int2name': name,
            'logits': torch.from_numpy(logits).float(),
            'input_ids': torch.from_numpy(input_ids).long(),
            'comparE': torch.from_numpy(comparE).float(),
            'label': torch.argmax(torch.from_numpy(logits)),
            # 'hidden_states': torch.from_numpy(hidden_states).float(),
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
        # hidden_states = [sample['hidden_states'] for sample in batch]
        # len_hidden_states = torch.tensor([len(sample['hidden_states'][0]) for sample in batch]).long()
        # all_layer = []
        # for layer in range(len(self.hidden_state_layers)):
        #     layer_hidden = [x[layer] for x in hidden_states]
        #     layer_hidden = pad_sequence(layer_hidden, batch_first=True, padding_value=0)
        #     all_layer.append(layer_hidden.unsqueeze(1))
        # hidden_states = torch.cat(all_layer, dim=1)
        

        return {
            'int2name': int2name,
            'logits': logits,
            'comparE': comparE,
            'len_comparE': len_comparE,
            'label': label,
            'input_ids': input_ids,
            'mask': mask
            # 'hidden_states': hidden_states,
            # 'len_hidden_states': len_hidden_states,
        }

if __name__ == '__main__':
    class test:
        data_type = "name"
    
    opt = test()
    a = MovieV2Dataset(opt, 'trn')
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

    from transformers import BertTokenizer, BertPreTrainedModel, BertModel
    import torch.nn as nn
    class BertClassifier(BertPreTrainedModel):
        def __init__(self, config, num_classes, embd_method): # 
            super().__init__(config)
            self.num_labels = num_classes
            self.embd_method = embd_method
            if self.embd_method not in ['cls', 'mean', 'max']:
                raise NotImplementedError('Only [cls, mean, max] embd_method is supported, \
                    but got', config.embd_method)

            self.bert = BertModel(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.cls_layer = nn.Linear(config.hidden_size, self.num_labels)
            self.init_weights()
        
        def forward(self, input_ids, attention_mask):
            # Feed the input to Bert model to obtain contextualized representations
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = outputs.last_hidden_state
            cls_token = outputs.pooler_output
            hidden_states = outputs.hidden_states
            # using different embed method
            if self.embd_method == 'cls':
                cls_reps = cls_token
            elif self.embd_method == 'mean':
                cls_reps = torch.mean(last_hidden, dim=1)
            elif self.embd_method == 'max':
                cls_reps = torch.max(last_hidden, dim=1)[0]
            
            cls_reps = self.dropout(cls_reps)
            logits = self.cls_layer(cls_reps)
            return logits, hidden_states

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = [tokenizer._convert_id_to_token(x.item()) for x in collatn['input_ids'][0]]
    print(collatn['input_ids'][0])
    print(tokens)
    print(collatn['mask'][0])
    print('----------')
    print(collatn['logits'])
    model = BertClassifier.from_pretrained('/data4/lrc/movie_dataset/pretrained/bert_movie_model', num_classes=5, embd_method='max')
    logits, _ = model(collatn['input_ids'], collatn['mask'])
    print(logits)
    # from tqdm import tqdm
    # for data in tqdm(a):
    #     pass