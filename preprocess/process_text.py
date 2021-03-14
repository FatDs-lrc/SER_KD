import os
import os.path as osp
import pandas as pd
import re
import random

def combine(file1, file2, save_file):
    content1 = open(file1).readlines()
    content2 = open(file2).readlines()
    content = content1 + content2
    ret = []
    partern = re.compile(r'[A-Za-z]',re.S)
    for line in content:
        utt, _ = line.strip().split('\t')
        if re.findall(partern, utt) and len(utt) > 1:
            ret.append(line)
        else:
            print(utt)
        
    fout = open(save_file, 'w')
    fout.writelines(ret)

def combine_emotionX_XED():
    emotionX_dir = '/data6/lrc/SER_KD/datasets/emotionX/processed_frd'
    XED_dir = '/data6/lrc/SER_KD/datasets/tool_man_text/xed_text'
    save_dir = '/data6/lrc/EmotionXED/combined'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    for set_name in ['trn', 'val', 'tst']:
        trn_e = osp.join(emotionX_dir, set_name + '.tsv')
        trn_x = osp.join(XED_dir, set_name + '.tsv')
        save_file = osp.join(save_dir, set_name + '.tsv')
        combine(trn_e, trn_x, save_file)

def gather_text_by_label(csv):
    ret = {}
    df = pd.read_csv(csv, header=None, encoding='utf8', delimiter='\t')
    for _, row in df.iterrows():
        sentence = row[0]
        label = row[1]
        if ret.get(label, None) is None:
            ret[label] = set(sentence)
        else:
            ret[label].add(sentence)
    return ret

def filter_text_label(text_set, sample_num):
    text_set = random.sample(text_set, sample_num)
    return text_set

def filter_texts(csv_path, save_path):
    text_label_set = gather_text_by_label(csv_path)
    if 'trn' in csv_path:
        text_label_set[0] = filter_text_label(text_label_set[0], 3500)
    elif 'val' in csv_path:
        text_label_set[0] = filter_text_label(text_label_set[0], 500)
    else:
        text_label_set[0] = filter_text_label(text_label_set[0], 1000)
    lines = []
    for k, v in text_label_set.items():
        for _v in v:
            lines.append(f'{_v}\t{k}\n')
    fout = open(save_path, 'w', encoding='utf8')
    print(len(lines))
    fout.writelines(lines)

if __name__ == '__main__':
    # combine_emotionX_XED()
    # ret = gather_text_by_label('/data6/lrc/EmotionXED/combined/tst.tsv')
    # for key in sorted(list(ret.keys())):
    #     print(key, len(ret[key]))
    '''
    RAW:
      trn   val     tst
    0 9469  1244    2747
    1 3126  421     814
    2 2678  378     770
    3 2115  303     585
    4 3151  499     936
    5 1903  300     536

    FILTERED:
      trn   val     tst
    0 3500  500     1000
    1 3126  421     814
    2 2678  378     770
    3 2115  303     585
    4 3151  499     936
    5 1903  300     536
    '''
    filter_texts('/data6/lrc/EmotionXED/combined/trn.tsv', '/data6/lrc/EmotionXED/filtered/trn.tsv')
    filter_texts('/data6/lrc/EmotionXED/combined/val.tsv', '/data6/lrc/EmotionXED/filtered/val.tsv')
    filter_texts('/data6/lrc/EmotionXED/combined/tst.tsv', '/data6/lrc/EmotionXED/filtered/tst.tsv')