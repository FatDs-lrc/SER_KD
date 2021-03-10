import os
import os.path as osp
import json
import pandas as pd

def read_data(json_file):
    label_map = {'neutral': 0, 'joy': 1, 'surprise': 2, 'sadness': 3, 'anger': 4, 'fear': 5}
    data = json.load(open(json_file))
    sentences = []
    labels = []
    for dialog in data:
        for sentence_pack in dialog:
            label = sentence_pack['emotion']
            sentence = sentence_pack['utterance']
            if label in label_map.keys():
                sentences.append(sentence)
                labels.append(label_map[label])
        
    assert len(sentences) == len(labels)
    return sentences, labels

def save_utt_label(utts, labels, save_path):
    f = open(save_path, 'w')
    for utt, label in zip(utts, labels):
        f.write(f'{utt}\t{label}\n')
    f.close()

def process_emotionX():
    ## train:
    frd_trn_utt, frd_trn_label = read_data('/data6/lrc/SER_KD/datasets/emotionX/raw/friends.train.json')
    push_trn_utt, push_trn_label = read_data('/data6/lrc/SER_KD/datasets/emotionX/raw/emotionpush.train.json')
    trn_utt = frd_trn_utt + push_trn_utt
    trn_label = frd_trn_label + push_trn_label

    ## val
    frd_val_utt, frd_val_label = read_data('/data6/lrc/SER_KD/datasets/emotionX/raw/friends.dev.json')
    push_val_utt, push_val_label = read_data('/data6/lrc/SER_KD/datasets/emotionX/raw/emotionpush.dev.json')
    val_utt = frd_val_utt + push_val_utt
    val_label = frd_val_label + push_val_label

    ## test
    frd_tst_utt, frd_tst_label = read_data('/data6/lrc/SER_KD/datasets/emotionX/raw/friends.test.json')
    push_tst_utt, push_tst_label = read_data('/data6/lrc/SER_KD/datasets/emotionX/raw/emotionpush.test.json')
    tst_utt = frd_tst_utt + push_tst_utt
    tst_label = frd_tst_label + push_tst_label

    save_dir = '/data6/lrc/SER_KD/datasets/emotionX/processed'
    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    save_utt_label(trn_utt, trn_label, osp.join(save_dir, 'trn.tsv'))
    save_utt_label(val_utt, val_label, osp.join(save_dir, 'val.tsv'))
    save_utt_label(tst_utt, tst_label, osp.join(save_dir, 'tst.tsv'))

def process_emotionX_frd():
    ## train:
    frd_trn_utt, frd_trn_label = read_data('/data6/lrc/SER_KD/datasets/emotionX/raw/friends.train.json')

    ## val
    frd_val_utt, frd_val_label = read_data('/data6/lrc/SER_KD/datasets/emotionX/raw/friends.dev.json')

    ## test
    frd_tst_utt, frd_tst_label = read_data('/data6/lrc/SER_KD/datasets/emotionX/raw/friends.test.json')

    save_dir = '/data6/lrc/SER_KD/datasets/emotionX/processed_frd'
    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    save_utt_label(frd_trn_utt, frd_trn_label, osp.join(save_dir, 'trn.tsv'))
    save_utt_label(frd_val_utt, frd_val_label, osp.join(save_dir, 'val.tsv'))
    save_utt_label(frd_tst_utt, frd_tst_label, osp.join(save_dir, 'tst.tsv'))

# process_emotionX()
process_emotionX_frd()