import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from train_dataset_classifier import load_datasets_processed
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale


def _plot_clusters(data, colors, cluster_labels, labels, ax_handle, title, annotate=False):
    for i, color in enumerate(colors):
        indices = np.where(cluster_labels == labels[i])
        X = data[indices]
        ax_handle.scatter(X[:, 0], X[:, 1], color=color, label=labels[i])
        if annotate:
            offset = np.random.uniform(-0.2, 0.2)
            while np.isclose(offset, 0, rtol=1e-02):
                offset = np.random.uniform(-0.2, 0.2)
            # simply annotate the first sample
            ax_handle.annotate(labels[i], (X[0, 0]+offset, X[0, 1]+offset), color=color, bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.03', alpha=0.5))
    ax_handle.title.set_text(title)
    return ax_handle


def _save_estimator(estimator, path):
    with open(path, 'wb') as f:
        pickle.dump(estimator, f)
        print("estimator pickle saved to: {}".format(path))


parser = argparse.ArgumentParser('k-means-clustering')
parser.add_argument('--dataset', type=str, default='/home/autodl/processed_datasets/1e4_meta')  # or '/home/autodl/processed_datasets/1e4_combined'
parser.add_argument('--save_estimator', type=bool, default=True)  # stored under dataset
parser.add_argument('--n_samples', type=int, default=10000)

parser.add_argument('--figures', dest='figures', action="store_true")
parser.add_argument('--no_figures', dest='figures', action="store_false")
parser.set_defaults(figures=False)
args = parser.parse_args()


cfg = {}
annotate = True
meta_features_path = args.dataset

cfg['proc_dataset_dir'] = meta_features_path
cfg["datasets"] = ['Chucky', 'Decal', 'Hammer', 'Hmdb51', 'Katze', 'Kreatur', 'Munster', 'Pedro', 'SMv2', 'Ucf101',
                         'binary_alpha_digits', 'caltech101', 'caltech_birds2010', 'caltech_birds2011', 'cats_vs_dogs', 'cifar100',
                         'cifar10', 'coil100', 'colorectal_histology', 'deep_weeds', 'emnist', 'eurosat', 'fashion_mnist',
                         'horses_or_humans', 'kmnist', 'mnist', 'oxford_flowers102']

n_samples_to_use = args.n_samples
dataset_list = load_datasets_processed(cfg, cfg["datasets"])
n_clusters = len(dataset_list)
print("using data: {}".format(args.dataset))
print("using n_samples: {}".format(args.n_samples))
print("{} datasets loaded".format(n_clusters))

X_train = [ds[0].dataset[:n_samples_to_use].numpy() for ds in dataset_list]  # 0 -> for now, only use train data for clustering
n_feature_dimensions = X_train[0].shape[1]
X_train = np.concatenate(X_train, axis=0)
X_labels = [ds[2] for ds in dataset_list]

#X_test = [ds[1].dataset.numpy() for ds in dataset_list]
#X_test = np.concatenate(X_test, axis=0)

""" normalization (for now: only scale to [0,1] to preserve shape and outliers) """
X_train = minmax_scale(X_train, axis=0)  # axis=0 -> scale each feature independently (not sample)

""" fit k-means clustering """
kmeans_estimator = KMeans(n_clusters=n_clusters, random_state=0, n_jobs=4).fit(X_train)

if args.save_estimator:
    file_name = "kmeans_estimator_{}_dimfeatures_{}_samples.pickle".format(n_feature_dimensions, n_samples_to_use)
    _save_estimator(estimator=kmeans_estimator, path=os.path.join(meta_features_path, file_name))

""" predict the cluster index for each sample """
X_indices = kmeans_estimator.predict(X_train)
X_labels_cluster_order = np.asarray([X_labels[i] for i in X_indices])

X_labels_unique, idx = np.unique(X_labels_cluster_order, return_index=True)
X_labels_unique = X_labels_unique[np.argsort(idx)]

if args.figures:
    """ reduce dimensionality of original data for plotting """
    print("generating TSNE plot")
    X_train_tsne_embedding = TSNE(n_components=2).fit_transform(X_train)
    print("generating PCA plot")
    X_train_pca_embedding = PCA(n_components=2).fit_transform(X_train)

    """ assign the cluster indices to dim-reduced data """
    fig = plt.figure(figsize=(12, 12))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(X_labels_unique)))

    ax1 = plt.subplot(211)
    ax1 = _plot_clusters(data=X_train_tsne_embedding, colors=colors, cluster_labels=X_labels_cluster_order, labels=X_labels_unique, ax_handle=ax1, annotate=annotate, title="t-SNE")
    #ax1.legend(prop={'size': 7}, loc="upper right")


    ax2 = plt.subplot(212)
    ax2 = _plot_clusters(data=X_train_pca_embedding, colors=colors, cluster_labels=X_labels_cluster_order, labels=X_labels_unique, ax_handle=ax2, annotate=annotate, title="PCA")
    #ax2.legend(prop={'size': 7}, loc="upper right")

    plot_title = "Datasets clustered with k-means \n({}-dim. meta-features, {} samples per dataset, feature scaling to [0,1])".format(n_feature_dimensions, n_samples_to_use)
    fig.suptitle(plot_title, fontsize=14)
    file_path = os.path.join(meta_features_path, "kmeans_{}_dimfeatures_{}_samples.png".format(n_feature_dimensions, n_samples_to_use))
    plt.savefig(file_path)
    print("file saved to {}".format(file_path))
    plt.show()


