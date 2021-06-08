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
    w1 = np.reshape(w, (w.shape[0], 1))
    wt = np.transpose(w1)
    p2 = np.inner(wt, X)
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
    arg1 = prd(w, X, y)
    numerator = np.multiply(w, X)
    denominator = np.ones((X.shape[0],)) + np.exp(arg1)
    arg2 = np.divide(numerator, denominator)
    grad = np.mean(arg2)
    
    return(grad)

# Create two blobs
N = 300
X, y = make_blobs(n_samples=N, centers=2, cluster_std=1, n_features=2, random_state=2)

# change labels 0 to -1
y[y==0] = -1

print("X.shape =", X.shape, "  y.shape =", y.shape)

w = np.ones((2,))
res = prd(w, X, y)