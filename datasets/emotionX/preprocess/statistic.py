import os.path as osp
import json
from functools import reduce
from collections import Counter

def get_all_utterance():
    utt = []
    labels = []
    root = osp.join(osp.dirname(osp.abspath(__file__)), '../', 'raw')
    # load trn
    trn_push = json.load(open(osp.join(root, 'emotionpush.train.json')))
    trn_frnd = json.load(open(osp.join(root, 'friends.train.json')))
    trn = reduce(lambda x, y: x+y, trn_push + trn_frnd)
    # load val
    val_push = json.load(open(osp.join(root, 'emotionpush.dev.json')))
    val_frnd = json.load(open(osp.join(root, 'friends.dev.json')))
    val = reduce(lambda x, y: x+y, val_push + val_frnd)
    # load test
    tst_push = json.load(open(osp.join(root, 'emotionpush.test.json')))
    tst_frnd = json.load(open(osp.join(root, 'friends.test.json')))
    tst = reduce(lambda x, y: x+y, tst_push + tst_frnd)
    # print infos
    print('Trn:', len(trn), 'Val:', len(val), 'Tst:', len(tst))
    # statis labels:
    trn_labels = [x['emotion'] for x in trn]
    val_labels = [x['emotion'] for x in val]
    tst_labels = [x['emotion'] for x in tst]
    print('Trn labels:', Counter(trn_labels))
    print('Val labels:', Counter(val_labels))
    print('Tst labels:', Counter(tst_labels))

if __name__ == "__main__":
    get_all_utterance()