# -*- coding: utf-8 -*-
"""
Aluno: Ciro B Rosa
No USP: 2320769
E-mail: ciro.rosa@alumni.usp.br
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(X_train_ori, y_train_ori), (X_test_ori, y_test_ori) = mnist.load_data()

print(X_train_ori.shape, y_train_ori.shape)
print(X_test_ori.shape, y_test_ori.shape)

labels = ["%s"%i for i in range(10)]

unique, counts = np.unique(y_train_ori, return_counts=True)
uniquet, countst = np.unique(y_test_ori, return_counts=True)

fig, ax = plt.subplots()
rects1 = ax.bar(unique - 0.2, counts, 0.25, label='Train')
rects2 = ax.bar(unique + 0.2, countst, 0.25, label='Test')
ax.legend()
ax.set_xticks(unique)
ax.set_xticklabels(labels)

plt.title('MNIST classes')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

fig, ax = plt.subplots(2, 3, figsize = (9, 6))

for i in range(6):
    ax[i//3, i%3].imshow(X_train_ori[i], cmap='gray')
    ax[i//3, i%3].axis('off')
    ax[i//3, i%3].set_title("Class: %d"%y_train_ori[i])
    
plt.show()

# Reduce the image size to its half 
X_train = np.array([image[::2, 1::2] for image in X_train_ori])
X_test  = np.array([image[::2, 1::2] for image in X_test_ori])

y_train = y_train_ori
y_test = y_test_ori

fig, ax = plt.subplots(2, 3, figsize = (9, 6))

for i in range(6):
    ax[i//3, i%3].imshow(X_train[i], cmap='gray')
    ax[i//3, i%3].axis('off')
    ax[i//3, i%3].set_title("Class: %d"%y_train_ori[i])
    
plt.show()

X_train = (X_train/255.0).astype('float32').reshape((60000,14*14))
X_test = (X_test/255.0).astype('float32').reshape((10000,14*14))

print(X_train.dtype)
print(X_test.dtype)

print("\nShape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)

print("\nMinimum value in X_train:", np.amin(X_train))
print("Maximum value in X_train:", np.amax(X_train))

print("\nMinimum value in X_test:", np.amin(X_test))
print("Maximum value in X_test:", np.amax(X_test))



# 1. Dataset preparation
# Split the original trainset in Dtrain / Dval (70 / 30)
from sklearn.model_selection import train_test_split

seed = 42   # fix seed to make results repeatable
X_Dtrain, X_Dval, y_Dtrain, y_Dval = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.30,
                                                      random_state=seed)

print("\nShape of X_Dtrain: ", X_Dtrain.shape)
print("Shape of X_Dval: ", X_Dval.shape)

# check for stratification in Dtrain and Dval
unique_t, counts_t = np.unique(y_Dtrain, return_counts=True)
unique_v, counts_v = np.unique(y_Dval, return_counts=True)

fig, ax = plt.subplots()
rects1 = ax.bar(unique_t - 0.2, counts_t, 0.25, label='Dtrain')
rects2 = ax.bar(unique_v + 0.2, counts_v, 0.25, label='Dval')
ax.legend()
ax.set_xticks(unique_t)
ax.set_xticklabels(labels)

plt.title('MNIST classes')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

# Dtrain split in 70 / 30
X_Strain, X_Sval, y_Strain, y_Sval = train_test_split(X_Dtrain,
                                                      y_Dtrain,
                                                      test_size=0.30,
                                                      random_state=seed)
print("\nShape of X_Strain: ", X_Strain.shape)
print("Shape of X_Sval: ", X_Sval.shape)


# 2. Training, evaluating and selecting models
# Dtrain (Strain, Sval) only

# libraries
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from sklearn.model_selection import GridSearchCV
import pandas as pd


# printing of performance results
def print_report(y_true, y_hat):
    cm = confusion_matrix(y_true, y_hat, labels = unique_t)
    print("\n", cm)
    
    cr = classification_report(y_true, y_hat, labels = unique_t, digits=3)
    print("\n", cr)
    
    f1 = f1_score(y_true, y_hat, labels = unique_t, average = "micro")
    print("f1-score: ", format(f1, ".3f"))
    
    return


"""
# logistic regression model
logistic_regression = LogisticRegression(max_iter=400)
logistic_regression.fit(X_Strain, y_Strain)
y_hat_log = logistic_regression.predict(X_Sval)

print_report(y_Sval, y_hat_log)
"""

# logistic regression optimization
parameters = {'C': [1.0, 0.75, 0.5, 0.25]}
logistic_regression = LogisticRegression(max_iter=400)
clf = GridSearchCV(logistic_regression, parameters, scoring='f1_micro')
clf = clf.fit(X_Strain, y_Strain)
y_hat_log = clf.predict(X_Sval)

print_report(y_Sval, y_hat_log)
print('best parameters: ', clf.best_params_)
df_log = pd.DataFrame(clf.cv_results_)   # usar para debug do gridsearch

"""
# neural network model
mlp_classifier = MLPClassifier()
mlp_classifier.fit(X_Strain, y_Strain)
y_hat_mlp = mlp_classifier.predict(X_Sval)

print_report(y_Sval, y_hat_mlp)

# neural network optimization
parameters = {'hidden_layer_sizes': [80, 100],
              'solver': ['lbfgs', 'adam'],
              'learning_rate': ['constant', 'adaptive']}
mlp_classifier = MLPClassifier()
clf = GridSearchCV(mlp_classifier, parameters, scoring='f1_micro')
clf = clf.fit(X_Strain, y_Strain)
y_hat_mlp = clf.predict(X_Sval)

print_report(y_Sval, y_hat_mlp)
print('best parameters: ', clf.best_params_)
df_mlp = pd.DataFrame(clf.cv_results_)   # usar para debug do gridsearch


# SVM model

# PCA

# returns the number of elements where its cumulative variance accounts
# for up to 95% of the total variance
def n_components_pca(v):
    v = -np.sort(-v)
    v = np.cumsum(v)
    v = v <= 0.95
    n = np.sum(v)
    
    return n

# optipmizing pca
pca = PCA(svd_solver='randomized', whiten=True)
pca = pca.fit(X_Strain)
n_pca = n_components_pca(pca.explained_variance_ratio_)
print("n_components definido como: ", n_pca)

# final pca
pca = PCA(n_components = n_pca, svd_solver='randomized', whiten=True)
pca = pca.fit(X_Strain)
X_Strain_pca = pca.transform(X_Strain)
X_Sval_pca = pca.transform(X_Sval)

# SVM
svm_classifier = SVC()
svm_classifier = svm_classifier.fit(X_Strain_pca, y_Strain)
y_hat_svm = svm_classifier.predict(X_Sval_pca)

print_report(y_Sval, y_hat_svm)

# SVM optimization
parameters = {'C': [1.0, 0.75, 0.5, 0.25],
              'kernel': ['rbf', 'poly', 'sigmoid'],
              'gamma': ['scale', 'auto']}
svm_classifier = SVC()
clf = GridSearchCV(svm_classifier, parameters, scoring='f1_micro')
clf = clf.fit(X_Strain_pca, y_Strain)
y_hat_svm = clf.predict(X_Sval_pca)

print_report(y_Sval, y_hat_svm)
print('best parameters: ', clf.best_params_)
df_svm = pd.DataFrame(clf.cv_results_)   # usar para debug do gridsearch
"""
