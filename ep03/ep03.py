# All imports

import numpy as np
import pandas as pd
import seaborn as sb

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.datasets import mnist

def prd(w, X, y):
    """
    computes the product of y.wt.X
    return res: shape=(N, 1)
    """
    Xt = np.transpose(X)
    p2 = np.dot(w, Xt)
    res = np.multiply(y, p2)
    
    return(res)


# 1.1. Cross-entropy loss
def cross_entropy_loss(w, X, y):
    """
    Computes the loss (equation 1)
    :param w: weight vector
    :type: np.ndarray(shape=(1+d, ))
    :param X: design matrix
    :type X: np.ndarray(shape=(N, 1+d))
    :param y: class labels
    :type y: np.ndarray(shape=(N, ))
    :return loss: loss (equation 1)
    :rtype: float
    """    
    arg1 = prd(w, X, y)
    arg2 = np.log(1 + np.exp(-arg1))
    loss = np.mean(arg2)
    
    return(loss)


# 1.2. Gradient of the cross-entropy loss
def cross_entropy_gradient(w, X, y):
    """
    Computes the gradient of the loss function (equation 2)
    :param w: weight vector
    :type: np.ndarray(shape=(1+d, ))
    :param X: design matrix
    :type X: np.ndarray(shape=(N, 1+d))
    :param y: class labels
    :type y: np.ndarray(shape=(N, ))
    :return grad: gradient (equation 2)
    :rtype: float
    """
    N = X.shape[0]
    
    numerator = np.dot(w, np.transpose(X))

    arg1 = prd(w, X, y)
    denominator = np.ones((N,)) + np.exp(arg1)
    arg2 = np.divide(numerator, denominator)
    grad = -np.mean(arg2)
    
    return(grad)


# 1.3 Logistic regression training
def train_logistic(X, y, learning_rate = 1e-1, w0 = None,\
                        num_iterations = 1000, return_history = False):
    """
    Computes the weight vector applying the gradient descent technique
    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: class label
    :type y: np.ndarray(shape=(N, ))
    :return: weight vector
    :rtype: np.ndarray(shape=(1+d, ))
    :return: the history of loss values (optional)
    :rtype: list of float
    """    
    
    N = X.shape[0]
    d = X.shape[1]
    ones = np.ones((N,1))
    X_one = np.hstack((ones, X))
    
    if w0 == None:
        w0 = np.random.normal(loc = 0, scale = 1, size = d)

    






# Create two blobs
N = 300
X, y = make_blobs(n_samples=N, centers=2, cluster_std=1, n_features=2, random_state=2)

# change labels 0 to -1
y[y==0] = -1

print("X.shape =", X.shape, "  y.shape =", y.shape)

w = np.ones((2,))
res = prd(w, X, y)
grad = cross_entropy_gradient(w,X,y)
train_logistic(X, y)