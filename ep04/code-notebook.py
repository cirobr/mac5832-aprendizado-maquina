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
# Dividir o trainset original em Dtrain / Dval (70 / 30)
from sklearn.model_selection import train_test_split

seed = 42
X_Dtrain, X_Dval, y_Dtrain, y_Dval = train_test_split(X_train,
                                                    y_train,
                                                    test_size=0.30,
                                                    random_state=seed)

print("\nShape of X_Dtrain: ", X_Dtrain.shape)
print("Shape of X_Dval: ", X_Dval.shape)
print("Shape of X_test: ", X_test.shape)

# Verificar a correta estratificação de Dtrain e Dval
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



# 2. Training, evaluating and selecting models

# bibliotecas
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# criar instâncias dos modelos
logistic_regression = LogisticRegression()

# criar funções para medida de mérito dos modelos classificadores
def accuracy(estimator, X, y):
    estimator.fit(X, y)
    y_hat = estimator.predict(X)
    acc = accuracy_score(y, y_hat)
    return acc


# 2.1 Training a logistic regression model
#res_log = accuracy(logistic_regression, X_Dtrain, y_Dtrain)
#print("\nLogistic Regression score:", "{:.3f}".format(res_log))

logistic_regression.fit(X_Dtrain, y_Dtrain)   # estudar, está falhando
y_hat = logistic_regression.predict(X_Dtrain)
acc = confusion_matrix(y_Dtrain, y_hat, labels = labels)




# 2.2 Optimizing the logistic regression model




# 2.3 Training a NN model




# 2.4 Optimizing a NN model




# 2.5 Training a SVM model




# 2.6 Optimizing a SVM model

