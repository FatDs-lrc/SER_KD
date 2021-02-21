import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

"""对S型曲线数据的降维和可视化"""
def load_data(modality, cv):
    # root = '/data2/lrc/Iemocap_feature/multi_fusion_reps/{}/{}'.format(modality, cv)
    root = '/data2/lrc/Iemocap_feature/early_fusion_reps/{}/{}'.format(modality, cv)
    trn = np.load(os.path.join(root, 'trn.npy'))
    val = np.load(os.path.join(root, 'val.npy'))
    tst = np.load(os.path.join(root, 'tst.npy'))
    data = np.concatenate([trn, val, tst], axis=0)
    val_tst_data = np.concatenate([val, tst], axis=0)
    return val_tst_data

def load_AE_data(model_type, cv):
    root = '/data2/lrc/Iemocap_feature/simple_AE_analysis/{}/{}'.format(model_type, cv)
    trn = np.load(os.path.join(root, 'trn_recon.npy'))
    val = np.load(os.path.join(root, 'val_recon.npy'))
    tst = np.load(os.path.join(root, 'tst_recon.npy'))
    data = np.concatenate([trn, val, tst], axis=0)
    val_tst_data = np.concatenate([val, tst], axis=0)
    return val_tst_data

def load_latent_data(cv):
    # root = '/data2/lrc/Iemocap_feature/simple_map_analysis/A2L/{}'.format(cv)
    # root = '/data2/lrc/Iemocap_feature/simple_AE_analysis/L2A/{}'.format(cv)
    # trn = np.load(os.path.join(root, 'trn_latent.npy'))
    # val = np.load(os.path.join(root, 'val_latent.npy'))
    # tst = np.load(os.path.join(root, 'tst_latent.npy'))

    root = '/data2/lrc/Iemocap_feature/test_latent_0416/{}'.format(cv)
    trn = np.load(os.path.join(root, 'trn.npy'))
    val = np.load(os.path.join(root, 'val.npy'))
    tst = np.load(os.path.join(root, 'tst.npy'))
    data = np.concatenate([trn, val, tst], axis=0)
    val_tst_data = np.concatenate([val, tst], axis=0)
    return val_tst_data

def load_label(cv):
    label_root = '/data2/lrc/Iemocap_feature/multi_fusion_reps/target/{}'.format(cv)
    trn_label = np.load(os.path.join(label_root, 'trn_label.npy'))
    val_label = np.load(os.path.join(label_root, 'val_label.npy'))
    tst_label = np.load(os.path.join(label_root, 'tst_label.npy'))
    val_tst_label = np.concatenate([val_label, tst_label], axis=0)
    return val_tst_label

def plot_modality(cv):
    # cv = sys.argv[1]
    # cv = int(cv)
    A = load_data('A', cv)
    V = load_data('V', cv)
    L = load_data('L', cv)
    labels = load_label(cv)
    labels = np.concatenate([labels, labels, labels], axis=0)
    lengthA = len(A)
    lengthV = len(V)
    lengthL = len(L)
    colors = ['red', 'blue', 'green', 'orange', 'deepskyblue', 'lightgreen']
    colors_label = ['red', 'blue', 'green', 'yellow']
    total_data = np.concatenate([A,V,L], axis=0)
    print(total_data.shape)
    print(labels.shape)
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # 训练模型
    y = ts.fit_transform(total_data)
    # 画各个模态的图
    A_tsne = y[:lengthA]
    V_tsne = y[lengthA: lengthA+lengthV]
    L_tsne = y[lengthA+lengthV: ]
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.scatter(A_tsne[:, 0], A_tsne[:, 1], s=1, c=colors[0], cmap=plt.cm.Spectral)
    plt.scatter(V_tsne[:, 0], V_tsne[:, 1], s=1, c=colors[1], cmap=plt.cm.Spectral)
    plt.scatter(L_tsne[:, 0], L_tsne[:, 1], s=1, c=colors[2], cmap=plt.cm.Spectral)
    plt.legend(['A', 'V', 'L'])

    # 画对应的label
    cat0_data = y[np.where(labels==0)]
    cat1_data = y[np.where(labels==1)]
    cat2_data = y[np.where(labels==2)]
    cat3_data = y[np.where(labels==3)]
    plt.subplot(122)
    plt.scatter(cat0_data[:, 0], cat0_data[:, 1], s=1, c=colors_label[0], cmap=plt.cm.Spectral)
    plt.scatter(cat1_data[:, 0], cat1_data[:, 1], s=1, c=colors_label[1], cmap=plt.cm.Spectral)
    plt.scatter(cat2_data[:, 0], cat2_data[:, 1], s=1, c=colors_label[2], cmap=plt.cm.Spectral)
    plt.scatter(cat3_data[:, 0], cat3_data[:, 1], s=1, c=colors_label[3], cmap=plt.cm.Spectral)
    plt.legend(['ang', 'hap', 'neu', 'sad'])
    # 显示图像
    # plt.show()
    plt.savefig("tsne/{}.png".format(cv))

def plot_AE(cv):
    # L2A
    A = load_data('A', cv)
    V = load_data('V', cv)
    L = load_data('L', cv)
    L2A = load_AE_data('L2A', cv)
    labels = load_label(cv)
    labels = np.concatenate([labels, labels, labels], axis=0)
    lengthA = len(A)
    lengthL = len(L)
    lengthL2A = len(L2A)
    colors = ['red', 'green', 'orange']
    colors_label = ['red', 'blue', 'green', 'yellow']

    total_data = np.concatenate([A,L,L2A], axis=0)
    print(total_data.shape)

    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # 训练模型
    y = ts.fit_transform(total_data)
    # 画各个模态的图
    A_tsne = y[:lengthA]
    L_tsne = y[lengthA: lengthA+lengthL]
    L2A_tsne = y[lengthA+lengthL: ]

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.scatter(A_tsne[:, 0], A_tsne[:, 1], s=1, c=colors[0], cmap=plt.cm.Spectral)
    plt.scatter(L_tsne[:, 0], L_tsne[:, 1], s=1, c=colors[1], cmap=plt.cm.Spectral)
    plt.scatter(L2A_tsne[:, 0], L2A_tsne[:, 1], s=1, c=colors[2], cmap=plt.cm.Spectral)
    plt.legend(['A', 'L', 'L2A'])

    cat0_data = y[np.where(labels==0)]
    cat1_data = y[np.where(labels==1)]
    cat2_data = y[np.where(labels==2)]
    cat3_data = y[np.where(labels==3)]
    plt.subplot(122)
    plt.scatter(cat0_data[:, 0], cat0_data[:, 1], s=1, c=colors_label[0], cmap=plt.cm.Spectral)
    plt.scatter(cat1_data[:, 0], cat1_data[:, 1], s=1, c=colors_label[1], cmap=plt.cm.Spectral)
    plt.scatter(cat2_data[:, 0], cat2_data[:, 1], s=1, c=colors_label[2], cmap=plt.cm.Spectral)
    plt.scatter(cat3_data[:, 0], cat3_data[:, 1], s=1, c=colors_label[3], cmap=plt.cm.Spectral)
    plt.legend(['ang', 'hap', 'neu', 'sad'])
    plt.savefig("tsne/AE/{}.png".format(cv))


def plot_map(cv):
    A = load_data('A', cv)
    V = load_data('V', cv)
    L = load_data('L', cv)
    latent = load_latent_data(cv)
    labels_raw = load_label(cv)
    labels = np.concatenate([labels_raw, labels_raw, labels_raw], axis=0)
    lengthA = len(A)
    lengthV = len(V)
    lengthL = len(L)
    colors = ['red', 'green', 'orange']
    colors_label = ['red', 'blue', 'green', 'yellow']
    total_data = np.concatenate([A,V,L], axis=0)
    print(total_data.shape)

    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # 训练模型
    y = ts.fit_transform(total_data)
    # 画各个模态的图
    A_tsne = y[:lengthA]
    V_tsne = y[lengthA: lengthA+lengthV]
    L_tsne = y[lengthA+lengthV: ]
    # 画AL
    plt.figure(figsize=(12, 5))
    plt.subplot(221)
    plt.scatter(A_tsne[:, 0], A_tsne[:, 1], s=1, c=colors[0], cmap=plt.cm.Spectral)
    plt.scatter(V_tsne[:, 0], V_tsne[:, 1], s=1, c=colors[1], cmap=plt.cm.Spectral)
    plt.scatter(L_tsne[:, 0], L_tsne[:, 1], s=1, c=colors[2], cmap=plt.cm.Spectral)
    plt.legend(['A', 'V', 'L'])
    # label
    cat0_data = y[np.where(labels==0)]
    cat1_data = y[np.where(labels==1)]
    cat2_data = y[np.where(labels==2)]
    cat3_data = y[np.where(labels==3)]
    plt.subplot(222)
    plt.scatter(cat0_data[:, 0], cat0_data[:, 1], s=1, c=colors_label[0], cmap=plt.cm.Spectral)
    plt.scatter(cat1_data[:, 0], cat1_data[:, 1], s=1, c=colors_label[1], cmap=plt.cm.Spectral)
    plt.scatter(cat2_data[:, 0], cat2_data[:, 1], s=1, c=colors_label[2], cmap=plt.cm.Spectral)
    plt.scatter(cat3_data[:, 0], cat3_data[:, 1], s=1, c=colors_label[3], cmap=plt.cm.Spectral)
    plt.legend(['ang', 'hap', 'neu', 'sad'])

    # 画 latent
    print('latent tsne')
    y = ts.fit_transform(latent)
    plt.subplot(223)
    plt.scatter(y[:, 0], y[:, 1], s=1, c=colors[0], cmap=plt.cm.Spectral)
    plt.legend(['latent'])
    # latent label
    cat0_data = y[np.where(labels_raw==0)]
    cat1_data = y[np.where(labels_raw==1)]
    cat2_data = y[np.where(labels_raw==2)]
    cat3_data = y[np.where(labels_raw==3)]
    plt.subplot(224)
    plt.scatter(cat0_data[:, 0], cat0_data[:, 1], s=1, c=colors_label[0], cmap=plt.cm.Spectral)
    plt.scatter(cat1_data[:, 0], cat1_data[:, 1], s=1, c=colors_label[1], cmap=plt.cm.Spectral)
    plt.scatter(cat2_data[:, 0], cat2_data[:, 1], s=1, c=colors_label[2], cmap=plt.cm.Spectral)
    plt.scatter(cat3_data[:, 0], cat3_data[:, 1], s=1, c=colors_label[3], cmap=plt.cm.Spectral)
    plt.legend(['ang', 'hap', 'neu', 'sad'])
    plt.savefig("tsne/new0416/{}.png".format(cv))


def load_teacher_mix(cv):
    root = '/data2/lrc/Iemocap_feature/early_fusion_reps/fusion/{}'.format(cv)
    trn = np.load(os.path.join(root, 'trn.npy'))
    val = np.load(os.path.join(root, 'val.npy'))
    tst = np.load(os.path.join(root, 'tst.npy'))
    data = np.concatenate([trn, val, tst], axis=0)
    val_tst_data = np.concatenate([val, tst], axis=0)

    label_root = '/data2/lrc/Iemocap_feature/early_fusion_reps/target/{}'.format(cv)
    trn_label = np.load(os.path.join(label_root, 'trn_label.npy'))
    val_label = np.load(os.path.join(label_root, 'val_label.npy'))
    tst_label = np.load(os.path.join(label_root, 'tst_label.npy'))
    val_tst_label = np.concatenate([val_label, tst_label], axis=0)
    print(val_tst_label.shape)
    return val_tst_data, val_tst_label

def load_latent_0416(cv):
    root = '/data2/lrc/Iemocap_feature/test_latent_0416/{}'.format(cv)
    trn = np.load(os.path.join(root, 'trn.npy'))
    val = np.load(os.path.join(root, 'val.npy'))
    tst = np.load(os.path.join(root, 'tst.npy'))
    data = np.concatenate([trn, val, tst], axis=0)
    val_tst_data = np.concatenate([val, tst], axis=0)

    label_root = '/data2/lrc/Iemocap_feature/test_latent_0416/{}'.format(cv)
    trn_label = np.load(os.path.join(label_root, 'trn_label.npy'))
    val_label = np.load(os.path.join(label_root, 'val_label.npy'))
    tst_label = np.load(os.path.join(label_root, 'tst_label.npy'))
    val_tst_label = np.concatenate([val_label, tst_label], axis=0)
    print(val_tst_label.shape)
    return val_tst_data, val_tst_label

def plot_new_0416(cv):
    print('teacher emb tsne')
    ef, label = load_teacher_mix(cv)
    colors = ['red', 'green', 'orange']
    colors_label = ['red', 'blue', 'green', 'yellow']
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # 训练模型
    y = ts.fit_transform(ef)
    # 画AL
    plt.figure(figsize=(12, 5))
    plt.subplot(221)
    plt.scatter(y[:, 0], y[:, 1], s=1, c=colors[2], cmap=plt.cm.Spectral)
    plt.legend(['teacher emb'])
    # label
    cat0_data = y[np.where(label==0)]
    cat1_data = y[np.where(label==1)]
    cat2_data = y[np.where(label==2)]
    cat3_data = y[np.where(label==3)]
    plt.subplot(222)
    plt.scatter(cat0_data[:, 0], cat0_data[:, 1], s=1, c=colors_label[0], cmap=plt.cm.Spectral)
    plt.scatter(cat1_data[:, 0], cat1_data[:, 1], s=1, c=colors_label[1], cmap=plt.cm.Spectral)
    plt.scatter(cat2_data[:, 0], cat2_data[:, 1], s=1, c=colors_label[2], cmap=plt.cm.Spectral)
    plt.scatter(cat3_data[:, 0], cat3_data[:, 1], s=1, c=colors_label[3], cmap=plt.cm.Spectral)
    plt.legend(['ang', 'hap', 'neu', 'sad'])

    # 画 latent
    print('latent tsne')
    latent, label = load_latent_0416(cv)
    y = ts.fit_transform(latent)
    plt.subplot(223)
    plt.scatter(y[:, 0], y[:, 1], s=1, c=colors[0], cmap=plt.cm.Spectral)
    plt.legend(['latent'])
    # latent label
    cat0_data = y[np.where(label==0)]
    cat1_data = y[np.where(label==1)]
    cat2_data = y[np.where(label==2)]
    cat3_data = y[np.where(label==3)]
    plt.subplot(224)
    plt.scatter(cat0_data[:, 0], cat0_data[:, 1], s=1, c=colors_label[0], cmap=plt.cm.Spectral)
    plt.scatter(cat1_data[:, 0], cat1_data[:, 1], s=1, c=colors_label[1], cmap=plt.cm.Spectral)
    plt.scatter(cat2_data[:, 0], cat2_data[:, 1], s=1, c=colors_label[2], cmap=plt.cm.Spectral)
    plt.scatter(cat3_data[:, 0], cat3_data[:, 1], s=1, c=colors_label[3], cmap=plt.cm.Spectral)
    plt.legend(['ang', 'hap', 'neu', 'sad'])
    plt.savefig("tsne/new0416/{}.png".format(cv))

if __name__ == '__main__':
    # for i in range(1,11):
    #     plot_AE(1)
    # plot_map(1)
    plot_new_0416(1)