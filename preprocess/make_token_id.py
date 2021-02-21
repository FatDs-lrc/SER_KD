import os
import h5py
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm

class IEMOCAP_text(object):
    def __init__(self, data_root):
        self.data_root = data_root
        self.dialog_record = set()
        self.lookup = {}
    
    def __getitem__(self, utt_id):
        ses_id = utt_id[4]
        dialog_id = '_'.join(utt_id.split('_')[:-1])
        if dialog_id not in self.dialog_record:
            file_path = f'{self.data_root}/Session{ses_id}/dialog/transcriptions/{dialog_id}.txt'
            content = open(file_path).readlines()
            for line in content:
                if not line.startswith('Ses'):
                    continue
                _utt_id = line.split()[0]
                transcripts = ' '.join(line.split()[2:])
                self.lookup[_utt_id] = transcripts
        return self.lookup[utt_id]

def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label

# def bert_tokenize(tokenizer, text):
#     ids = []
#     for word in text.strip().split():
#         ws = tokenizer.tokenize(word)
#         if not ws:
#             # some special char
#             continue
#         ids.extend(tokenizer.convert_tokens_to_ids(ws))
#     return ids

def make_all_token_ids(config):
    text_lookup = IEMOCAP_text(config['data_root'])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    trn_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'trn')
    val_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'tst')
    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
    tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    all_h5f = h5py.File(os.path.join(config['feature_root'], 'bert_token_ids', 'all.h5'), 'w')
    for utt_id in tqdm(all_utt_ids):
        text = text_lookup[utt_id]
        token_ids = tokenizer.encode(text)
        all_h5f[utt_id] = np.array(token_ids)

def padding_to_fixlen(data, length, padding_value=0):
    if len(data) >= length:
        ret = data[:length]
    else:
        ret = np.concatenate([data, np.ones(length-len(data))*padding_value], axis=0)
    return ret

def split_cv(config):
    max_len = 22
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    padding_value = tokenizer._convert_token_to_id('[PAD]')
    all_ft = h5py.File(os.path.join(config['feature_root'], 'bert_token_ids', 'all.h5'), "r")
    for cv in range(1, 11):
        save_dir = os.path.join(config['feature_root'], 'bert_token_ids', str(cv))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for set_name in ['trn', 'val', 'tst']:
            int2name, _ = get_trn_val_tst(config['target_root'], cv, set_name)
            int2name = list(map(lambda x: x[0].decode(), int2name))
            fts = []
            for utt_id in int2name:
                ft = all_ft[utt_id][()]
                ft = padding_to_fixlen(ft, max_len, padding_value)
                fts.append(ft)
            fts = np.array(fts)
            print(f'{cv} {set_name} {fts.shape}')
            np.save(os.path.join(save_dir, f'{set_name}.npy'), fts)

if __name__ == '__main__':
    config = {
        'data_root': '/data3/lrc/IEMOCAP_full_release',
        'face_root': '/data3/lrc/IEMOCAP',
        'feature_root': '/data7/lrc/IEMOCAP_features_npy/feature',
        'target_root': '/data7/lrc/IEMOCAP_features_npy/target'
    }
    # make_all_token_ids(config)
    split_cv(config)