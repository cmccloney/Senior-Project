from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from catboost import Pool, CatBoostClassifier
import random
import os
from sklearn.externals import joblib as jl
from PolyHandler import PolyHandler

samples_dir = "/home/lizard/483_Landsat_project/samples/"


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

sub_sample_names  = ["{}A_poly_samples.npy".format(samples_dir)]
dst_files = ["{}A_RF".format(samples_dir)]
for i in range(1):
    sample = np.load(sub_sample_names[i])
    print("\nZZZZZZZZZZZZZZZZZZZZZZZZZ\n{}".format(sample.shape))
    x = sample[:, :-1, 0]
    y = sample[:, -1, 0]
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=23)
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)
    model = CatBoostClassifier(iterations=2100,learning_rate=.18, loss_function="MultiClass",eval_metric="TotalF1", random_state=23).fit(train_pool)
    y_pred = model.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, title='Confusion matrix, without normalization', classes=model.classes_)
    plt.savefig("Mesa_cat.pdf")

