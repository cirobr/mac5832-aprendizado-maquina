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


### 1.1. Cross-entropy loss
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


### 1.2. Gradient of the cross-entropy loss
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


### 1.3 Logistic regression training
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

    cross_entropy_loss_history = np.empty(shape=(0,), dtype=float)
    for t in range(num_iterations):
        
        if return_history:
            cross_entropy_loss_history = np.append(cross_entropy_loss_history, \
                                                   cross_entropy_loss(w0, X_one, y))

        gt = cross_entropy_gradient(w0, X_one, y)
        vt = -np.copy(gt)
        wt = w0 + learning_rate * vt
        w0 = np.copy(wt)
        
        # observação: considerar melhoria no critério de término do loop
        # Mostafa, p.96
        
    if return_history:
        res = (wt, cross_entropy_loss_history)
    else:
        res = wt
    
    return(res)


### 1.4. Logistic regression prediction
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_logistic(X, w):
    """
    Computes the logistic regression prediction
    :param X: design matrix
    :type X: np.ndarray(shape=(N,d))
    :param w: weight vector
    :rtype: np.ndarray(shape=(1+d,))
    :return: predicted classes 
    :rtype: np.ndarray(shape=(N,))
    """
    
    wt = np.transpose(w)
    h = sigmoid(np.dot(wt, X))
    
    return (h)


# 2.1. Generate two blobs of points
# Create two blobs
N = 300
X, y = make_blobs(n_samples=N, centers=2, cluster_std=1, n_features=2, random_state=2)

# change labels 0 to -1
y[y==0] = -1

print("X.shape =", X.shape, "  y.shape =", y.shape)


# 2.2. Let's plot the blobs of points
fig = plt.figure(figsize=(6,6))

# plot negatives in red
plt.scatter(X[y==-1,0], \
            X[y==-1,1], \
            alpha = 0.5,\
            c = 'red')

# and positives in blue
plt.scatter(x=X[y==1,0], \
            y=X[y==1,1], \
            alpha = 0.5, \
            c = 'blue')

P=+1
N=-1
legend_elements = [ Line2D([0], [0], marker='o', color='r',\
                    label='Class %d'%N, markerfacecolor='r',\
                    markersize=10),\
                    Line2D([0], [0], marker='o', color='b',\
                    label='Class %d'%P, markerfacecolor='b',\
                    markersize=10) ]

plt.legend(handles=legend_elements, loc='best')
plt.show()


### 2.3. Let's train the linear regressor and plot the loss curve
np.random.seed(567)

# ==> Replace the right hand side below with a call to the
# train_logistic() function defined above. Use parameter return_history=True

#w_logistic, loss = np.array([0,0,0]), [0]
w_logistic, loss = train_logistic(X, y)

# ==> Your code insert ends here

print()
print("Final weight:\n", w_logistic)
print()
print("Final loss:\n", loss[-1])

plt.figure(figsize = (12, 8))
plt.plot(loss)
plt.xlabel('Iteration #')
plt.ylabel('Cross Entropy Loss')
plt.show()