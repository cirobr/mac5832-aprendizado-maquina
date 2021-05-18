# some imports
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


# An auxiliary function
def get_housing_prices_data(N, verbose=True):
    """
    Generates artificial linear data,
    where x = square meter, y = house price

    :param N: data set size
    :type N: int
    
    :param verbose: param to control print
    :type verbose: bool
    :return: design matrix, regression targets
    :rtype: np.array, np.array
    """
    cond = False
    while not cond:
        x = np.linspace(90, 1200, N)
        gamma = np.random.normal(30, 10, x.size)
        y = 50 * x + gamma * 400
        x = x.astype("float32")
        x = x.reshape((x.shape[0], 1))
        y = y.astype("float32")
        y = y.reshape((y.shape[0], 1))
        cond = min(y) > 0
        
    xmean, xsdt, xmax, xmin = np.mean(x), np.std(x), np.max(x), np.min(x)
    ymean, ysdt, ymax, ymin = np.mean(y), np.std(y), np.max(y), np.min(y)
    if verbose:
        print("\nX shape = {}".format(x.shape))
        print("y shape = {}\n".format(y.shape))
        print("X: mean {}, sdt {:.2f}, max {:.2f}, min {:.2f}".format(xmean,
                                                               xsdt,
                                                               xmax,
                                                               xmin))
        print("y: mean {:.2f}, sdt {:.2f}, max {:.2f}, min {:.2f}".format(ymean,
                                                                 ysdt,
                                                                 ymax,
                                                                 ymin))
    return x, y

# Another auxiliary function
def plot_points_regression(x,
                           y,
                           title,
                           xlabel,
                           ylabel,
                           prediction=None,
                           legend=False,
                           r_squared=None,
                           position=(90, 100)):
    """
    Plots the data points and the prediction,
    if there is one.

    :param x: design matrix
    :type x: np.array
    :param y: regression targets
    :type y: np.array
    :param title: plot's title
    :type title: str
    :param xlabel: x axis label
    :type xlabel: str
    :param ylabel: y axis label
    :type ylabel: str
    :param prediction: model's prediction
    :type prediction: np.array
    :param legend: param to control print legends
    :type legend: bool
    :param r_squared: r^2 value
    :type r_squared: float
    :param position: text position
    :type position: tuple
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    line1, = ax.plot(x, y, 'bo', label='Real data')
    if prediction is not None:
        line2, = ax.plot(x, prediction, 'r', label='Predicted data')
        if legend:
            plt.legend(handles=[line1, line2], loc=2)
        ax.set_title(title,
                 fontsize=20,
                 fontweight='bold')
    if r_squared is not None:
        bbox_props = dict(boxstyle="square,pad=0.3",
                          fc="white", ec="black", lw=0.2)
        t = ax.text(position[0], position[1], "$R^2 ={:.4f}$".format(r_squared),
                    size=15, bbox=bbox_props)

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    plt.show()

X, y = get_housing_prices_data(N=100)
"""
plot_points_regression(X,
                       y,
                       title='Real estate prices prediction',
                       xlabel="m\u00b2",
                       ylabel='$')
"""
def normal_equation_weights(X, y):
    """
    Calculates the weights of a linear function using the normal equation method.
    You should add into X a new column with 1s.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :return: weight vector
    :rtype: np.ndarray(shape=(d+1, 1))
    """
    
    # START OF YOUR CODE:
    #raise NotImplementedError("Function normal_equation_weights() is not implemented")
    
    import numpy as np
    from numpy.linalg import inv
    N = X.shape[0]
    #d = X.shape[1]
    X_0 = np.ones((N,1))                # false X_0 coordinate
    X_til = np.append(X_0, X, axis=1)   # insert false coordinate to X
    X_til = X_til.astype("float32")
    Xt = X_til.T
    X_aux = inv(np.dot(Xt, X_til))
    X_cross = np.dot(X_aux, Xt)
    w = np.dot(X_cross, y)
    
    return(w)
    # END OF YOUR CODE
    
def normal_equation_prediction(X, w):
    """
    Calculates the prediction over a set of observations X using the linear function
    characterized by the weight vector w.
    You should add into X a new column with 1s.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param w: weight vector
    :type w: np.ndarray(shape=(d+1, 1))
    :param y_hat: regression prediction
    :type y_hat: np.ndarray(shape=(N, 1))
    """
    
    # START OF YOUR CODE:
    #raise NotImplementedError("Function normal_equation_prediction() is not implemented")
    
    N = X.shape[0]
    #d = X.shape[1]
    
    X_0 = np.ones((N,1))                # false X_0 coordinate
    X_til = np.append(X_0, X, axis=1)   # insert false coordinate to X
    X_til = X_til.astype("float32")

    y_hat = np.multiply(w.T, X_til)     # multiplica cada elemento de w.T por coluna em X_till
    yh0 = y_hat[:,0].reshape((N, 1))
    yh1 = y_hat[:,1:]
    y_hat = yh0 + yh1
    
    return(y_hat)

    # END OF YOUR CODE

# load the dataset
df = pd.read_csv('QT1data.csv')
f = df.iloc[:,5].str.isnumeric()
df = df[f]
print(df.head())

# Our target variable is the weight
y = df.pop('Weight').values
y = y.reshape((y.shape[0], 1))
print(y.T)

feature_cols = ['Height']
X = df.loc[:, feature_cols]
print(X.shape)


# START OF YOUR CODE:
w = normal_equation_weights(X, y)
print("Estimated w =\n", w)

prediction = normal_equation_prediction(X, w)
print(prediction.T)

r_2 = r2_score(y_true=y,
               y_pred=prediction)
print(r_2)

plot_points_regression(X,
                       y,
                       title='Weight prediction',
                       xlabel="height (cm)",
                       ylabel='weight (Kg)',
                       prediction=prediction,
                       legend=True,
                       r_squared=r_2)

# END OF YOUR CODE
