### ep01-1


import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Create a numpy array with N points
xmin = ymin = -1
xmax =ymax = 2

X = np.asarray([[1.3, -0.2],[0,0],[0,1],[1,0],[1,1]])
print("Shape of array X: ", X.shape)
N = X.shape[0]
print("Number of examples: ", N)

# add a left column with 1's into X -- X extended,
# that way Xi has the same number of elements of the weight array
def add_column_of_ones(X):
    return np.hstack(( np.ones((X.shape[0],1)), X ) )
Xe = add_column_of_ones(X)
print("Shape of array Xe: ", Xe.shape)

# define a target weight vector
w_target = np.asarray([[0.5],[-1], [1]])
print("Shape of array w_target: ", w_target.shape)
print("Target weight array: \n", w_target)

# define y (class) values, based on the line defined by the target weight vector
y = np.sign(np.dot(Xe, w_target))
print("Shape of array y: ", y.shape)


# Plotting ...

# plot the line
a = -w_target[1] / w_target[2] # slope  -- we will have trouble if w_target[2]=0 ...
b = -w_target[0] / w_target[2] # intercept
x_l = np.linspace(xmin, xmax, 50)
y_l = a*x_l + b

plt.figure(figsize=(6,6))
plt.plot(x_l, y_l);
axes = plt.gca()
axes.set_xlim([xmin,xmax])
axes.set_ylim([xmin,ymax])

# Determine the colors of each of the examples
colors = ["blue" if y[i]==1 else "red" for i in range(N)]
print("color: ", colors)

# plot the examples
plt.scatter(X[:,0],X[:,1],c=colors)


### ep01-2


def plot_state(Xe,w,xmin=-1,xmax=2,ymin=-1,ymax=2):
    # compute yhat - prediction
    yhat = np.sign(np.dot(Xe,w))

    correct = np.where(y == yhat)[0]
    misclassified = np.where(y != yhat)[0]

    colors_o = ["blue" if y[i]==1 else "red" for i in correct]
    colors_x = ["blue" if y[i]==1 else "red" for i in misclassified]

    # plotting
    a = -w[1] / w[2] # slope
    b = -w[0] / w[2] # intercept
    x_l = np.linspace(-1, 2, 50)
    y_l = a*x_l + b

    plt.figure(figsize=(6,6))
    plt.plot(x_l, y_l);
    axes = plt.gca()
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])
    plt.title("Weight:   w0=%.2f    w1=%.2f    w2=%.2f" %(w[0],w[1],w[2]))
    plt.scatter(X[correct,0],X[correct,1],c=colors_o, marker='o');
    plt.scatter(X[misclassified,0],X[misclassified,1],c=colors_x, marker='x');

    
# Starting weight vector <---- change as you wish (as long as w0[2] != 0)
w0 = np.asarray([[-0.5], [1] , [1]])
   
plot_state(Xe,w0)


### ep01-3


def perceptron(Xe,y,w0,plot=False):
    """
    Parameters:
       Xe   : ndarray (N,d+1) - it already has the 1's in column 0
       y    : ndarray (N,1)
       w0   : ndarray (d+1,1) - the initial weight vector
       plot : If True, plot the state at the beginning of each iteration
       
    Returns:
       w : ndarray (d+1,1) - the final weight vector
    """
    
    # START OF YOUR CODE:
    w = np.array(w0)

    while True:
        yhat = np.sign(np.dot(Xe,w))
        misclassified = np.where(y != yhat)[0]
        if misclassified.shape[0] == 0:
            break
        
        # w(t+1) = w(t) + y(t) * x(t)
        yx1 = y[misclassified] * Xe[misclassified]
        yx2 = np.sum(yx1, axis = 0)
        yx3 = np.atleast_2d(yx2)
        yx = np.transpose(yx3)
        w = w + yx

        if plot:
            plot_state(Xe, w)

    # END YOUR CODE
    
    return w


# Test your function for w0, X and y as defined above
w0 = np.asarray([[-0.5], [1] , [1]]) # <---- you can change it (as long as w0[2] != 0)
print("Initial weight vector=\n", w0)

w = perceptron(Xe,y,w0,plot=True)


# Print the final weight vector and plot the final graph using the function plot_state.
# START OF YOUR CODE:
print("Final weight vector=\n", w)
plot_state(Xe, w)

# END YOUR CODE
