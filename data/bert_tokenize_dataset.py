import torch
import numpy as np
from data.base_dataset import BaseDataset
from transformers import AutoTokenizer

class IEMOCAP_text(object):
    def __init__(self, data_root):
        self.data_root = data_root
        self.dialog_record = set()
        self.lookup = {}
    
    def _register(self, dialog_id):
        ses_id = dialog_id[4]
        file_path = f'{self.data_root}/Session{ses_id}/dialog/transcriptions/{dialog_id}.txt'
        content = open(file_path).readlines()
        for line in content:
            if not line.startswith('Ses'):
                continue
            _utt_id = line.split()[0]
            transcripts = ' '.join(line.split()[2:])
            self.lookup[_utt_id] = transcripts
        self.dialog_record.add(dialog_id)

    def __getitem__(self, utt_id):
        dialog_id = '_'.join(utt_id.split('_')[:-1])
        if dialog_id not in self.dialog_record:
            self._register(dialog_id)
        return self.lookup[utt_id]

class BertTokenizeDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        return parser

    def __init__(self, opt, set_name): 
        super().__init__(opt)
        self.maxlen = 68
        # data_root = '/data3/lrc/IEMOCAP_full_release/'
        data_root = '/home/zzc/lrc/IEMOCAP_full_release/'
        self.text_lookup = IEMOCAP_text(data_root)
        self.tokenizer = AutoTokenizer.from_pretrained(opt.bert_type)
        # load label
        cvNo = opt.cvNo
        # label_path = "/data7/lrc/IEMOCAP_features_npy/target/{}/"
        label_path = "/home/zzc/lrc/IEMOCAP_features_npy/target/{}/" 
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_label.npy")
        self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(label_path.format(cvNo) + f"{set_name}_int2name.npy")
        self.int2name = [x[0].decode() for x in self.int2name]
        self.dialog_ids = set(['_'.join(x.split('_')[:-1]) for x in self.int2name])
        for dialog_id in self.dialog_ids:
            self.text_lookup._register(dialog_id)
       
        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):    
        # find label and raw text
        utt_id = self.int2name[index]
        sentence = self.text_lookup[utt_id]
        label = self.label[index]

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
        cvNo = 1
    
    dataset = BertTokenizeDataset(test, 'trn')
    input_ids, attn_mask, label = dataset[0]
    print(input_ids)
    print(attn_mask)
    print(label)