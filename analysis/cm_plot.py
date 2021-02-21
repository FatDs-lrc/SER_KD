import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import sys

_type = sys.argv[1]
title = _type
classes = ['ang', 'hap', 'neu', 'sad']
n_classes = len(classes)

# L
# cm = np.array([[ 67,  3,  10,   2],
#                 [  4, 112,  21,   9],
#                 [ 17,  40, 135,  21],
#                 [  8,  11, 20,  77]])

# recon
# cm = np.array([[ 61,   6,  10,  5],
#                 [ 10, 111,  18,   7],
#                 [ 14,  37, 119,  43],
#                 [  2,   6,  24,  84]])

# if _type == 'L':
#     cm = np.array([[ 62,   5,  13,   2],
#                 [  2, 121,  17,   6],
#                 [ 12,  60, 119,  22],
#                 [  6,  20,  16,  74]])

# elif _type == 'L_recon':
#     cm = np.array([[ 59,   6,  13,   4],
#                     [  3, 115,  22,   6],
#                     [ 13,  32, 128,  40],
#                     [  2,   3,  36,  75]])

# elif _type == 'A':
#     cm = np.array([[ 55,  15,   7,   5],
#                 [ 40,  47,  23,  36],
#                 [ 19,  43,  56,  95],
#                 [  3,   3,  10, 100]])

# seq cv1
# if _type == 'L':
#     cm = np.array([[109,   9,  22,   7],
#                 [  6, 100,  14,  12],
#                 [ 24,  33,  67,  47],
#                 [  4,  15,  14,  45]])
# elif _type == 'L_recon':
#     cm = np.array([[112,  15,  16,   4],
#                     [ 29,  93,   2,   8],
#                     [ 15,  50,  29,  77],
#                     [  0,  10,   3,  65]])
# seq cv2
if _type == 'L':
    cm = np.array([[ 67,   4,   9,   2],
                [  7, 107,  22,  10],
                [ 26,  53, 113,  21],
                [  9,  18,  15,  74]])
elif _type == 'L_recon':
    cm = np.array([[ 60,   5,   7,  10],
                    [ 27,  72,  17,  30],
                    [ 26,  25,  65,  97],
                    [  5,   1,   4, 106]])



cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Normalized confusion matrix")

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# ax.figure.colorbar(im, ax=ax)
# We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       # ... and label them with the respective list entries
    #    xticklabels=classes, yticklabels=classes,
    #    title=title,
    #    ylabel='True label',
    #    xlabel='Predicted label',
    #    fontsize=14
    )
ax.set_title(title, fontsize=16)
ax.set_xticklabels(classes, fontsize=12)
ax.set_yticklabels(classes, fontsize=12)
ax.set_xlabel('Predicted label', fontsize=14)
ax.set_ylabel('True label', fontsize=14)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(),rotation=60)

# Loop over data dimensions and create text annotations.
fmt = '.2f'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=14)
fig.tight_layout()

plt.savefig(f'analysis/cm/{_type}_seq_cv2.png')


