import torch
import os.path as osp
import numpy as np
import pandas as pd
from data.base_dataset import BaseDataset
from transformers import AutoTokenizer

class TextDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, default=1, \
                help='which cross validation set')
        parser.add_argument('--data_type', type=str, default='combined')
        parser.add_argument('--no_test', action='store_true')
        parser.add_argument('--no_val', action='store_true')
        return parser

    def __init__(self, opt, set_name): 
        super().__init__(opt)
        self.maxlen = 68
        data_root = '/data6/lrc/EmotionXED/{}'.format(opt.data_type)
        # data_root = '/home/zzc/lrc/EmotionXED/{}'.format(opt.data_type)
        self.tokenizer = AutoTokenizer.from_pretrained(opt.bert_type)   
        self.no_test = opt.no_test
        self.no_val = opt.no_val
        assert not(self.no_test and self.no_val)
        if self.no_test and set_name == 'trn':
            trn_data = pd.read_csv(osp.join(data_root, 'trn.tsv'), header=None, delimiter='\t')
            tst_data = pd.read_csv(osp.join(data_root, 'tst.tsv'), header=None, delimiter='\t')
            data = pd.concat([trn_data, tst_data])
        elif self.no_val and set_name == 'trn':
            trn_data = pd.read_csv(osp.join(data_root, 'trn.tsv'), header=None, delimiter='\t')
            val_data = pd.read_csv(osp.join(data_root, 'val.tsv'), header=None, delimiter='\t')
            data = pd.concat([trn_data, val_data])
        else:
            data = pd.read_csv(osp.join(data_root, f'{set_name}.tsv'), header=None, delimiter='\t')
        self.sentences = data[0].tolist()
        self.labels = data[1].tolist()
        
        print(f"TEXT dataset {set_name} created with total length: {len(self)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):    
        # find label and raw text
        sentence = self.sentences[index]
        label = self.labels[index]

        # Preprocess the text to be suitable for the transformer
        tokens = self.tokenizer.tokenize(sentence) 
        tokens = ['[CLS]'] + tokens + ['[SEP]'] 
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] 
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] 
        
        # Obtain the indices of the tokens in the BERT Vocabulary
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
        input_ids = torch.tensor(input_ids) 
        # print(tokens)
        # print(input_ids)
        # input()

        # Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'label': label
        }

if __name__ == '__main__':
    class test:
        data_type = 'combined'
        bert_type = 'bert-base-uncased'
    
    dataset = TextDataset(test, 'trn')
    input_ids, attn_mask, label = dataset[0].values()
    print(input_ids)
    print(attn_mask)
    print(label)