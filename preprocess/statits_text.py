import os
import os.path as osp
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

def statistic_tsv(tsv_file):
    df = pd.read_csv(tsv_file, delimiter='\t', header=None)
    df.columns = ['utt', 'label']
    ret = df['label'].value_counts(ascending=False)
    ret = ret.sort_index()
    print(ret)

def statis_length(sentences):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    lengths = []
    for sentence in tqdm(sentences):
        tokens = tokenizer.tokenize(sentence) 
        tokens = ['[CLS]'] + tokens + ['[SEP]']     
        lengths.append(len(tokens))

    lengths = sorted(lengths)
    print('MIN:', min(lengths))
    print('MAX:', max(lengths))
    print('MEAN: {:.2f}'.format(sum(lengths) / len(lengths)))
    print('50%:', lengths[len(lengths)//2])
    print('75%:', lengths[int(len(lengths)*0.75)])
    print('90%:', lengths[int(len(lengths)*0.9)])

def read_sentence(tsv_path):
    df = pd.read_csv(tsv_path, delimiter='\t', header=None)
    return df[0].tolist()

def get_all_sentences():
    root = '/data6/lrc/EmotionXED/combined'
    trn = osp.join(root, 'trn.tsv')
    val = osp.join(root, 'val.tsv')
    tst = osp.join(root, 'tst.tsv')
    trn = read_sentence(trn)
    val = read_sentence(val)
    tst = read_sentence(tst)
    all = trn + val + tst
    for x in all[:10]:
        print(x)
    statis_length(all)

if __name__ == '__main__':
    print('XED')
    statistic_tsv('/data6/lrc/EmotionXED/XED/trn.tsv')
    statistic_tsv('/data6/lrc/EmotionXED/XED/val.tsv')
    statistic_tsv('/data6/lrc/EmotionXED/XED/tst.tsv')

    # print('emotionX')
    # statistic_tsv('/data6/lrc/EmotionXED/emotionX_frd/trn.tsv')
    # statistic_tsv('/data6/lrc/EmotionXED/emotionX_frd/val.tsv')
    # statistic_tsv('/data6/lrc/EmotionXED/emotionX_frd/tst.tsv')

    print('combined')
    statistic_tsv('/data6/lrc/EmotionXED/combined/trn.tsv')
    statistic_tsv('/data6/lrc/EmotionXED/combined/val.tsv')
    statistic_tsv('/data6/lrc/EmotionXED/combined/tst.tsv')

    # get_all_sentences()