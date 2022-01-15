import itertools
import pickle

import matplotlib.pyplot as plt

import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from clustering.confusion_matrix import plot_confusion_matrix
from clustering.gmm_clusters import map_clusters_to_classes_by_majority, make_ellipses

print(__doc__)

colors = ['red', 'green', 'blue', 'brown', 'black', 'chocolate', 'darkblue', 'purple', 'orange']


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
            confusion=False, examples_per_class=1000, config=None):
    """
    Fits a GMM to the embeddings in name_to_embeddings where each name represents a dataset.
    """
    if last_principal_component_shown <= first_principal_component_shown:
        raise Exception('first PCA component must be smaller than the 2nd')
    
    name = config.name

    if not os.path.exists(name):
        os.makedirs(name)

    # Compute PCA
    if pca:
        pca = PCA(n_components=1 + last_principal_component_shown)
        #pca_transform = pca.fit(name_to_embeddings)[:, list(range(first_principal_component_shown, last_principal_component_shown + 1))]
        if not config.find_clusters_for_unseen: 
            pca_fitted = pca.fit(name_to_embeddings)
            filename = '{}/pca.pkl'.format(name)
            with open(filename, 'wb') as f:
                pickle.dump(pca_fitted, f)
                print("Saved PCA fitted")
        else:
            with open('{}/pca.pkl'.format(config.trained_gmm_path), 'rb') as f:
                pca_fitted = pickle.load(f)
                print("Loaded PCA fitted")
        
        pca_data = pca_fitted.transform(name_to_embeddings)[:, list(range(first_principal_component_shown, last_principal_component_shown +1))]        
    else:
        pca_data = name_to_embeddings
    
    pca_labels = []
    #print("We have {} classes".format(class_names))
    for i in range(len(class_names)):
        for j in range(examples_per_class):
            pca_labels.append(i)
    pca_labels = np.array(pca_labels)
    
    # Do not split the data - train=test=all (unsupervised evaluation)
    train_index = list(range(0, pca_data.shape[0]))
    print(pca_data.shape)
    X_train = pca_data[train_index]
    y_train = pca_labels[train_index]
    
    n_classes = len(np.unique(y_train))
    if clusters > 0:
        n_clusters = clusters
    else:
        n_clusters = n_classes
    print("We have {} classes.".format(n_clusters))

    # Can try GMMs using different types of covariances, we use full.
    estimators = {'full': GaussianMixture(n_components=n_clusters,
                                            covariance_type='full', max_iter=150, init_params='kmeans')}
    n_estimators = len(estimators)

    # Configure the plot
    if plot:
        main_plot = plt.figure(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.1, top=0.95, hspace=.15, wspace=.05, left=.09, right=.99)
    bic = []
    best_accuracy = 0
    for index, (name, estimator) in enumerate(estimators.items()):
        if config.find_clusters_for_unseen: suffix = 'inference'
        else: suffix = 'training'
        #name = config.name

        # train the GMM
        if config.find_clusters_for_unseen:
            with open('{}/gmm_estimator.pkl'.format(config.trained_gmm_path), 'rb') as f:
                estimator_used = pickle.load(f)
                print("Loaded trained GMM")
        else:
            
            estimator_used = estimator.fit(X_train)
            with open('{}/gmm_estimator.pkl'.format(config.name), 'wb') as f:
                pickle.dump(estimator_used, f)
                print("Saved trained GMM")
        #bic.append(estimator_used.bic(X_train))

        #print('full', 'spherical', 'diag', 'tied')
        #print(bic)
        
        class_names_clean = []

        for _name in class_names:
            _name = _name.split("_")
            if len(_name) > 1:
                class_names_clean.append(_name[1])
            else:
                class_names_clean.append(_name[0])

        # create the plots
        a = []
        if plot:
                #if not config.find_clusters_for_unseen:
                h = plt.subplot(1, 1, 1)
                # red, green, blue, yellow
                # Plot the train data with dots
                for n, domains in enumerate(class_names):
                    data = pca_data[[index for index in range(len(pca_labels)) if pca_labels[index] == n]]

                    ind = n % 9
                    if n < 5:
                        marker = 'o'
                    elif n < 10:
                        marker = 'x'
                    elif n < 15:
                        marker = "*"
                    elif n < 20:
                        marker = '.'
                    elif n < 25:
                        marker = 'v'
                    elif n < 30:
                        marker = '^'
                    elif n < 35:
                        marker = '+'
                    elif n < 40:
                        marker = '_'
                    elif n < 45:
                        marker = 's'
                    elif n < 50:
                        marker = 'D'
                    a.append(plt.scatter(data[:, 0], data[:, 1], s=20, marker=marker, color=colors[ind], alpha=0.3))

                plt.legend(a, class_names_clean, loc='lower right', bbox_to_anchor=(1.05, 1.0))
                plt.tight_layout()
        
        # predict the cluster ids for train
        y_train_pred = estimator_used.predict(X_train)
        from collections import Counter

        # map clusters to classes by majority of true class in cluster
        clusters_to_classes, classes_to_clusters, counter_clusters = map_clusters_to_classes_by_majority(y_train, y_train_pred)
        # plot confusion matrix, error analysis
        if confusion:
            y_pred_by_majority = np.array([clusters_to_classes[pred] for pred in y_train_pred])
            _, num_sequences_per_cluster, domain_x_percentage_in_each_cluster = plot_confusion_matrix(y_train,
                                                                                                      y_pred_by_majority,
                                                                                                      class_names_clean,
                                                                                                      title=header,
                                                                                                      max_size=
                                                                                                      examples_per_class,
                                                                                                      name=config.name)
        # Calculate the Purity metric
        count = 0
        #if not config.find_clusters_for_unseen:
        #    np.save('{}/internet_domain_percentage_in_each_cluster.npy'.format(config.name), domain_x_percentage_in_each_cluster)
        
        for i, pred in enumerate(y_train_pred):
            if clusters_to_classes[pred] == y_train[i]:
                count += 1
        train_accuracy = float(count) / len(y_train_pred) * 100

        if plot:
            make_ellipses(estimator_used, h, clusters_to_classes, colors)
            plt.xticks(())
            plt.yticks(())
            leg = plt.legend(scatterpoints=1, loc='best', prop=dict(size=5), bbox_to_anchor=(1, 0.8))
            for lh in leg.legendHandles:
                lh.set_alpha(1)
                lh._sizes = [30]

    if plot:
        plt.suptitle(header) 
        main_plot.savefig("{}/main.pdf".format(config.name), bbox_inches='tight')
        plt.show()

    # Now let's take the means and covariances of the gmms and cluster them hierarchically
    # One cluster per domain
    print("train_acc")
    print(train_accuracy)
    #return train_accuracy
    g_means, g_covariances = [], []
    ignored_clusters = []
    nonempty_clusters = []
    for n_cluster in range(n_clusters):
        #if num_sequences_per_cluster[n_cluster] == 0:
        if n_cluster not in classes_to_clusters.values():
            print("Ignoring cluster {} as it is empty.".format(n_cluster))
            ignored_clusters.append(n_cluster)
        else:
            nonempty_clusters.append(n_cluster)
            covariances = estimator_used.covariances_[n_cluster][:2, :2]
            means = np.array(estimator_used.means_[n_cluster, :2])
            g_means.append(means.transpose())
            g_covariances.append(covariances)
    match_old_new_clusters = {}
    cluster_id = 0
    for i in range(n_clusters):
        if i not in ignored_clusters:
            match_old_new_clusters[i] = cluster_id
            cluster_id += 1
    print("The old clusters correspond to the new ones as shown in dict (empty clusters deleted): \n {}".format(match_old_new_clusters))
    #print("---")
    domains_picked = []
    for cl in clusters_to_classes:
        if cl in ignored_clusters:
            continue
        domains_picked.append(clusters_to_classes[cl])
        print("Cluster {} has by majority domain {}".format(cl, clusters_to_classes[cl]))
    
    domains_not_picked = []
    for i in range(n_clusters):
        if i not in domains_picked:
            print("Domain {} is not a majority in any cluster".format(i))
            domains_not_picked.append(i)

    for dom in domains_not_picked:
        max_samples = 0 
        for cl, counter_cluster_i in enumerate(counter_clusters):
            if dom in counter_cluster_i.keys():
                if counter_cluster_i[dom] > max_samples and cl not in ignored_clusters:
                    max_samples = counter_cluster_i[dom]
                    assign_to_cluster = cl
                    unpicked_dom = dom
    
        if max_samples > 0:
            temp = clusters_to_classes[assign_to_cluster] 
            clusters_to_classes[assign_to_cluster] = []
            clusters_to_classes[assign_to_cluster].append(temp)
            clusters_to_classes[assign_to_cluster].append(unpicked_dom)
            print(clusters_to_classes[assign_to_cluster])
            print("Domain {} assigned to (old) cluster {}, it has {} samples that have max prob to go there".format(unpicked_dom,assign_to_cluster, max_samples))
    
    clusters_to_classes_new = {}
    for i in clusters_to_classes.keys():
        if i not in ignored_clusters:
            clusters_to_classes_new[i] = clusters_to_classes[i]
            
    clusters_to_classes_final_indices = {}
    for key, value in clusters_to_classes_new.items():
        key_final = match_old_new_clusters[key]
        clusters_to_classes_final_indices[key_final] = value
    
    classes_to_clusters_final = {}

    print("Cluster to domains final dictionary: {}".format(clusters_to_classes_final_indices))
    for key, value in clusters_to_classes_final_indices.items():
        print(value)
        if isinstance(value, list):
            print("MPIKA")
            for val in value:
                classes_to_clusters_final[str(val)] = key 
        else:
            classes_to_clusters_final[str(value)] = key
    print(classes_to_clusters_final)
    import json
    with open('{}/domain_to_cluster.json'.format(config.name), 'w') as f:
        json.dump(classes_to_clusters_final, f)
        print("saved domain-to-cluster in domain_to_cluster.json!")
   #print("-----")
    #print(counter_clusters)

    #print(sorted(domains_picked))
    kl_div_average_list = []

    for n in range(len(nonempty_clusters)):
        kl_div_nm= []
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
        ind += 1
        #print("cluster {} mostly has data from internet domain {}".format(n,clusters_to_classes[n]))

    main_plot = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.1, top=0.95, hspace=.15, wspace=.05, left=.09, right=.99)
    agg = AgglomerativeClustering(distance_threshold=0, linkage="average", affinity='precomputed', n_clusters=None)

    #agg = agg.fit(kl_div_array)
    if not config.find_clusters_for_unseen:
        agg_fitted = agg.fit(kl_div_array)
        with open("{}/agglomerativeclustering.pkl".format(config.name), 'wb') as f:
            pickle.dump(agg_fitted, f)
    else: 
        exit()
        with open("{}/agglomerativeclustering.pkl".format(config.trained_gmm_path), 'rb') as f:
            agg_fitted = pickle.load(f)

    ii = itertools.count(kl_div_array.shape[0])
    agg_clusters = [{'node_id': next(ii), 'left': x[0], 'right': x[1]} for x in agg_fitted.children_]

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
    for key, value in up_merge.items():
        print(" '{}': {},".format(key, value))

    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(agg_fitted,  truncate_mode='level', labels=labels,
                    leaf_font_size=9, leaf_rotation=0)

    main_plot.savefig("{}/main_dendrogram.pdf".format(config.name), bbox_inches='tight')

    plt.show()

    return train_accuracy
