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
import json
print(__doc__)

def inference_from_gmm(name_to_embeddings, class_names, first_principal_component_shown=0,
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

    with open('{}/ignored_clusters.json'.format(config.trained_gmm_path), 'r') as f:
        ignored_clusters = json.load(f)

    # Compute PCA
    if pca:
        pca = PCA(n_components=1 + last_principal_component_shown)
        #if not config.find_clusters_for_unseen: 
        if not config.trained_gmm_path:
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
    for i in range(len(class_names)):
        for j in range(examples_per_class):
            pca_labels.append(i)
    pca_labels = np.array(pca_labels)
    
    # Do not split the data - train=test=all (unsupervised evaluation)
    train_index = list(range(0, pca_data.shape[0]))
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

    best_accuracy = 0
    for index, (name, estimator) in enumerate(estimators.items()):

        # inference using trained GMM
        with open('{}/gmm_estimator.pkl'.format(config.trained_gmm_path), 'rb') as f:
            estimator_used = pickle.load(f)
            print("Loaded trained GMM")
            
        # predict the cluster ids for train
        y_train_pred = estimator_used.predict(X_train)
        from collections import Counter

        # map clusters to classes by majority of true class in cluster
        clusters_to_classes, classes_to_clusters, counter_clusters = map_clusters_to_classes_by_majority(y_train, y_train_pred)
        
        # Calculate the Purity metric
        count = 0
        for i, pred in enumerate(y_train_pred):
            if clusters_to_classes[pred] == y_train[i]:
                count += 1
        train_accuracy = float(count) / len(y_train_pred) * 100

    # data_samples of each domain (keys in dict) assigned to each cluster (key in inner dict)
    # with X samples in it (value in inner dict)
    data_samples_of_domain_in_cluster = {}
    for class_ind in range(len(class_names)):
        data_samples_of_domain_in_cluster[class_ind] = {}
    ind = 0
    for cluster_ind in clusters_to_classes.keys():
        for key, value in counter_clusters[ind].items():
            data_samples_of_domain_in_cluster[key][cluster_ind] = value
        ind = ind + 1

    domains_picked = []
    for cl in clusters_to_classes:
        domains_picked.append(clusters_to_classes[cl])
        #print("Cluster {} has by majority domain {}".format(cl, clusters_to_classes[cl]))
    
    domains_not_picked = []
    for i in range(n_clusters):
        if i not in domains_picked:
            #print("Domain {} is not a majority in any cluster".format(i))
            domains_not_picked.append(i)

    for domain in domains_not_picked:
        # get cluster with max samples of this domain in it
        cluster_picked = max(data_samples_of_domain_in_cluster[domain], key=data_samples_of_domain_in_cluster[domain].get)
        classes_to_clusters[domain] = cluster_picked
        
    for key, value in classes_to_clusters.items():
        classes_to_clusters[key] = int(value)
        
    ignored_clusters = list(ignored_clusters)
    classes_to_clusters_new = {}

    classes_to_clusters_without_ignored = {}
    for key,value in classes_to_clusters.items():
        if value not in ignored_clusters:
            classes_to_clusters_without_ignored[key] = value
        
        
    # Choose second path through the tree:
    for domain in domains_picked + domains_not_picked:    
        clusters_picked_list = []
        #we need to get the list of clusters (first one has most priority) that domain i should be assigned to (based on max samples allocation)
        for k in sorted(data_samples_of_domain_in_cluster[domain], key=data_samples_of_domain_in_cluster[domain].get, reverse=True):
            if k not in ignored_clusters:
                clusters_picked_list.append(k)
        if domain in classes_to_clusters_without_ignored.keys():
            first_cluster = classes_to_clusters_without_ignored[domain]
            if first_cluster != clusters_picked_list[0]:
                second_cluster = clusters_picked_list[0] 
            else:
                second_cluster = clusters_picked_list[1]
        else:
            first_cluster, second_cluster = clusters_picked_list[0], clusters_picked_list[1]
        
        #print("domain {}, first cluster {}, second cluster {}".format(domain,first_cluster,second_cluster))
        classes_to_clusters_new[domain] = []
        classes_to_clusters_new[domain].append(int(first_cluster))
        classes_to_clusters_new[domain].append(int(second_cluster))
    
    # Make keys strings to dump to json
    classes_to_clusters_for_json = {}
    for key, value in classes_to_clusters_new.items():
        classes_to_clusters_for_json[str(key)] = value

    #print("Domains to clusters assignment {}.".format(classes_to_clusters_new))
    
    with open('{}/match_old_new_clusters.json'.format(config.trained_gmm_path), 'r') as f:
        match_old_new = json.load(f)

    match_old_new_clusters = {}
    for key, value in match_old_new.items():
        match_old_new_clusters[int(key)] = value 

    classes_to_clusters_final_indices = {}
    for key, value in classes_to_clusters_for_json.items():
        classes_to_clusters_final_indices[str(key)] = []
        #print(key, value)
        for val in value:
            classes_to_clusters_final_indices[key].append(match_old_new_clusters[val])
    
    print("Domains to clusters with final indices assignment {}".format(classes_to_clusters_final_indices))
    with open('{}/domain_to_cluster.json'.format(config.name), 'w') as f:
        json.dump(classes_to_clusters_final_indices, f)
        print("saved domain-to-cluster in domain_to_cluster.json!")
       
