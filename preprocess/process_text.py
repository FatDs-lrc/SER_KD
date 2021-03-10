import os
import os.path as osp
import re

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

if __name__ == '__main__':
    combine_emotionX_XED()