import itertools
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import gensim

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from clustering.confusion_matrix import plot_confusion_matrix
from clustering.gmm_clusters import map_clusters_to_classes_by_majority, make_ellipses

print(__doc__)

colors = ['red', 'green', 'blue', "yellow", 'brown', 'black', 'chocolate', 'darkblue', 'purple', 'orange']


def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.

    - accepts stacks of means, but only one S0 and S1

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term = np.trace(iS1 @ S0)
    det_term = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N)


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    shc.dendrogram(linkage_matrix, **kwargs)


def fit_gmm_and_hierarchical(name_to_embeddings, class_names, first_principal_component_shown=0,
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
        clusters_to_classes, classes_to_clusters = map_clusters_to_classes_by_majority(y_train, y_train_pred)

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
            _, num_sequences_per_cluster, domain_x_percentage_in_each_cluster = plot_confusion_matrix(y_train,
                                                                                                      y_pred_by_majority,
                                                                                                      class_names,
                                                                                                      title=header,
                                                                                                      max_size=examples_per_class)

        # Calculate the Purity metric
        count = 0
        np.save('./internet_domain_percentage_in_each_cluster.npy', domain_x_percentage_in_each_cluster)
        for i, pred in enumerate(y_train_pred):
            # print("Sample {} from internet domain {} belongs to cluster {}.".format(i,
            #                                                                         pca_labels[i], clusters_to_classes[pred]))
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

    # Now let's take the means and covariances of the gmms and cluster them hierarchically
    # One cluster per domain

    g_means, g_covariances = [], []
    ignored_clusters = []
    nonempty_clusters = []
    for n in range(n_clusters):
        if num_sequences_per_cluster[n] == 0:
            print("Ignoring cluster {} as it is empty.".format(n))
            ignored_clusters.append(n)
        else:
            nonempty_clusters.append(n)
            covariances = estimator.covariances_[n][:2, :2]
            means = np.array(estimator.means_[n, :2])
            g_means.append(means.transpose())
            g_covariances.append(covariances)
    print(class_names)
    print(domain_x_percentage_in_each_cluster)
    # for i, row in enumerate(domain_x_percentage_in_each_cluster):
    #     for j, item in enumerate(row):
    #         print("{} % of domain {} belongs to cluster {}".format(item, class_names[i], j))

    kl_div_average_list = []

    for n in range(len(nonempty_clusters)):
        kl_div_nm = []
        kl_div_mn = []
        kl_div_average_per_cluster = []
        for m in range(len(nonempty_clusters)):
            if n != m:
                kl_div_nm.append((kl_mvn(g_means[n], g_covariances[n], g_means[m], g_covariances[m])))
                kl_div_mn.append((kl_mvn(g_means[m], g_covariances[m], g_means[n], g_covariances[n])))
            elif n == m:
                kl_div_nm.append(0)
                kl_div_mn.append(0)
        for i in range(len(nonempty_clusters)):
            kl_div_average_per_cluster.append((kl_div_nm[i] + kl_div_mn[i])/2)
        kl_div_average_list.append(kl_div_average_per_cluster)
    kl_div_array = np.array(kl_div_average_list)

    labels = []
    ind = 0 
    for n in sorted(list(clusters_to_classes.keys())):
        if n in ignored_clusters:
            continue
        labels.append(ind)
        ind+=1
        print("cluster {} mostly has data from internet domain {}".format(n,
                                                                          class_names[clusters_to_classes[n]]))

    main_plot = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.1, top=0.95, hspace=.15, wspace=.05, left=.09, right=.99)
    agg = AgglomerativeClustering(distance_threshold=0, linkage="average", affinity='precomputed', n_clusters=None)
    agg = agg.fit(kl_div_array)

    ii = itertools.count(kl_div_array.shape[0])
    agg_clusters = [{'node_id': next(ii), 'left': x[0], 'right': x[1]} for x in agg.children_]

    import copy
    n_points = kl_div_array.shape[0]
    members = {i: [i] for i in range(n_points)}
    for cluster in agg_clusters:
        node_id = cluster["node_id"]
        members[node_id] = copy.deepcopy(members[cluster["left"]])
        members[node_id].extend(copy.deepcopy(members[cluster["right"]]))

    on_split = {c["node_id"]: [c["left"], c["right"]] for c in agg_clusters}
    up_merge = {c["left"]: c["node_id"] for c in agg_clusters}
    up_merge.update({c["right"] : c["node_id"] for c in agg_clusters})
    #print(up_merge)
    #final = {}
    for key, value in up_merge.items():
        print(" '{}': {},".format(key, value))
    #    if isinstance(up_merge[key], np.integer):
    #        final[str(key)] = int(up_merge[key])
    #    elif isinstance(up_merge[key], np.floating):
    #        final[str(key)] = float(up_merge[key])
    #    elif isinstance(up_merge[key], np.ndarray):
    #        final[str(key)] = up_merge[key].tolist()
            
    import json
    
    #json.dump(final, open("./domain_dict.json", "w"))
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(agg,  truncate_mode='level', labels=labels,
                    leaf_font_size=12, leaf_rotation=0)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")

    main_plot.savefig("./main_dendrogram.pdf", bbox_inches='tight')

    plt.show()

    return best_accuracy
