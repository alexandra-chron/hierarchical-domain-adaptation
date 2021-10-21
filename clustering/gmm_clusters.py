from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import gensim

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from clustering.confusion_matrix import plot_confusion_matrix

print(__doc__)

colors = ['red', 'green', 'blue', "yellow", 'brown', 'black', 'chocolate', 'darkblue', 'purple', 'orange']


def make_ellipses(gmm, ax, clusters_to_classes, colors):
    """
    Adds Ellipses to ax according to the gmm clusters.
    """
    for n in sorted(list(clusters_to_classes.keys())):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        class_id = clusters_to_classes[n]
        class_id = class_id % 10
        class_color = colors[class_id]
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=class_color, linewidth=0)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.4)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')


def map_clusters_to_classes_by_majority(y_train, y_train_pred):
    """
    Maps clusters to classes by majority to compute the Purity metric.
    """
    cluster_to_class = {}
    class_to_cluster = {}
    for cluster in np.unique(y_train_pred):
        # run on indices where this is the cluster
        original_classes = []
        for i, pred in enumerate(y_train_pred):
            if pred == cluster:
                original_classes.append(y_train[i])
        # take majority
        cluster_to_class[cluster] = max(set(original_classes), key=original_classes.count)
        class_to_cluster[max(set(original_classes), key=original_classes.count)] = cluster
    return cluster_to_class, class_to_cluster


def fit_gmm(name_to_embeddings, class_names, first_principal_component_shown=0,
            last_principal_component_shown=1, clusters=5, header='', plot=True, pca=True,
            confusion=False, examples_per_class=1000):
    """
    Fits a GMM to the embeddings in name_to_embeddings where each name represents a dataset.
    """
    all_states = []
    all_sents = []
    num_classes = len(class_names)
    if last_principal_component_shown <= first_principal_component_shown:
        raise Exception('first PCA component must be smaller than the 2nd')

    # Concatenate the data to one matrix
    # for label in class_names:
    #     all_states.append(name_to_embeddings[label]['states'][0:examples_per_class])
    #     all_sents += name_to_embeddings[label]['sents']
    # concat_all_embs = np.concatenate(all_states)

    # Compute PCA
    if pca:
        pca = PCA(n_components=1 + last_principal_component_shown)
        pca_data = pca.fit_transform(name_to_embeddings)[:,
                   list(range(first_principal_component_shown, last_principal_component_shown + 1))]
    else:
        pca_data = name_to_embeddings

    pca_labels = []
    for i in range(len(class_names)):
        for j in range(examples_per_class):
            pca_labels.append(i)
    pca_labels = np.array(pca_labels)
    #     print(pca_labels)
    # Do not split the data - train=test=all (unsupervised evaluation)
    train_index = list(range(0, pca_data.shape[0]))
    test_index = list(range(0, pca_data.shape[0]))

    X_train = pca_data[train_index]
    y_train = pca_labels[train_index]
    X_test = pca_data[test_index]
    y_test = pca_labels[test_index]

    n_classes = len(np.unique(y_train))
    if clusters > 0:
        n_clusters = clusters
    else:
        n_clusters = n_classes

    # Can try GMMs using different types of covariances, we use full.
    estimators = {cov_type: GaussianMixture(n_components=n_clusters,
                                            covariance_type=cov_type, max_iter=150, random_state=0)
                  for cov_type in ['full']}  # 'spherical', 'diag', 'tied',

    n_estimators = len(estimators)

    # Configure the plot
    if plot:
        main_plot = plt.figure(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.1, top=0.95, hspace=.15, wspace=.05, left=.09, right=.99)

    best_accuracy = 0
    for index, (name, estimator) in enumerate(estimators.items()):

        # train the GMM
        estimator.fit(X_train)

        a = []
        # create the plots
        if plot:
            h = plt.subplot(1, 1, 1)
            # red, green, blue, yellow
            # Plot the train data with dots
            for n, domains in enumerate(class_names):
                data = pca_data[[index for index in range(len(pca_labels)) if pca_labels[index] == n]]

                ind = n % 10
                if n < 10:
                    marker = 'o'
                elif n < 20:
                    marker = 'x'
                elif n < 30:
                    marker = "*"
                elif n < 40:
                    marker = '.'
                elif n < 50:
                    marker = 'v'
                elif n < 60:
                    marker = '^'
                elif n < 70:
                    marker = '+'
                elif n < 80:
                    marker = '_'
                elif n < 90:
                    marker = 's'
                elif n < 100:
                    marker = 'D'
                a.append(plt.scatter(data[:, 0], data[:, 1], s=20, marker=marker, color=colors[ind], alpha=0.3))

            plt.legend(a, class_names, loc='lower right', bbox_to_anchor=(1.05, 1.0))
            plt.tight_layout()


        # predict the cluster ids for train
        y_train_pred = estimator.predict(X_train)

        # predict the cluster ids for test
        y_test_pred = estimator.predict(X_test)

        # map clusters to classes by majority of true class in cluster
        clusters_to_classes = map_clusters_to_classes_by_majority(y_train, y_train_pred)

        # plot confusion matrix, error analysis
        if confusion:
            from spacy.lang.en import English

            nlp = English()
            # Create a Tokenizer with the default settings for English
            # including punctuation rules and exceptions
            tokenizer = nlp.tokenizer
            digits_counter = defaultdict(int)
            digits_counter_pred = defaultdict(int)
            count_num_errors = 0
            count_errors = 0
            subs_prons = 0
            subs_erros = 0
            subs_prons_overall = 0
            sent_lens = []
            y_pred_by_majority = np.array([clusters_to_classes[pred] for pred in y_train_pred])
            plot_confusion_matrix(y_train, y_pred_by_majority, class_names, title=header)

        # Calculate the Purity metric
        count = 0
        for i, pred in enumerate(y_train_pred):
            if clusters_to_classes[pred] == y_train[i]:
                count += 1
        train_accuracy = float(count) / len(y_train_pred) * 100

        correct_count = 0
        for i, pred in enumerate(y_test_pred):
            if clusters_to_classes[pred] == y_test[i]:
                correct_count += 1
        test_accuracy = float(correct_count) / len(y_test_pred) * 100

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

        if plot:
            make_ellipses(estimator, h, clusters_to_classes, colors)
            plt.xticks(())
            plt.yticks(())
            leg = plt.legend(scatterpoints=1, loc='best', prop=dict(size=10), bbox_to_anchor=(1, 0.8))
            for lh in leg.legendHandles:
                lh.set_alpha(1)
                lh._sizes = [60]

    if plot:
        plt.suptitle(header)
        main_plot.savefig("./main.pdf", bbox_inches='tight')
        plt.show()

    return best_accuracy
