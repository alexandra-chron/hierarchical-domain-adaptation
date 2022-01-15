import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, max_size=None, name='new_clusters'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    uniq = unique_labels(y_true, y_pred)
    classes = np.array(classes)[uniq]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    def myfunction(x):
        return(100 * x/max_size)

    domain_x_percentage_in_each_cluster = np.apply_along_axis(myfunction, axis=1, arr=cm)
    #print(cm)

    #print("Each cluster has this many sequences:")
    num_sequences_per_cluster = np.sum(cm, axis=0)
    #print(num_sequences_per_cluster)
    # classes = [c.replace('_dev', '') for c in classes]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # Show all ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # And label them with the respective list entries
           xticklabels=[i for i in range(len(classes))], yticklabels=classes,
           ylabel='Internet Domain',
           xlabel='Cluster')

    # Size of Internet domain + cluster
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)

    # Rotate the tick labels and set their alignment.
    # Size of labels for X and Y
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor", fontsize=4)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", size=4)
    fig.tight_layout()
    fig.savefig("{}/main_confusion.pdf".format(name), bbox_inches='tight')
    return ax, num_sequences_per_cluster, domain_x_percentage_in_each_cluster
