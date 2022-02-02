import matplotlib as mpl

import numpy as np


print(__doc__)

colors = ['red', 'green', 'blue',  'brown', 'black', 'chocolate', 'darkblue', 'purple', 'orange']


def make_ellipses(gmm, ax, clusters_to_classes, colors):
    """
    Adds Ellipses to ax according to the gmm clusters.
    """
    print("Will print cluster and corresponding domain")
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
        #print(n, class_id) 
        #print("row {} of gmm ".format(n))
        class_id = class_id % 9
        
        class_color = colors[class_id]
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=class_color, linewidth=0)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.4)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')


def map_clusters_to_classes_by_majority(y_train, y_train_pred, ignored_clusters=None):
    """
    Maps clusters to classes by majority to compute the Purity metric.
    """
    cluster_to_class = {}
    class_to_cluster = {}
    counter_clusters = []
    for cluster in np.unique(y_train_pred):
        # run on indices where this is the cluster
        original_classes = []
        for i, pred in enumerate(y_train_pred):
            if pred == cluster:
                original_classes.append(y_train[i])
        if ignored_clusters is not None:
            if cluster not in ignored_clusters:
                # take the majority
                cluster_to_class[cluster] = max(set(original_classes), key=original_classes.count)
                from collections import Counter
                counter_clusters.append(Counter(original_classes))
                class_to_cluster[max(set(original_classes), key=original_classes.count)] = cluster
        else:
            cluster_to_class[cluster] = max(set(original_classes), key=original_classes.count)
            from collections import Counter
            counter_clusters.append(Counter(original_classes))
            class_to_cluster[max(set(original_classes), key=original_classes.count)] = cluster 
    
    return cluster_to_class, class_to_cluster, counter_clusters
