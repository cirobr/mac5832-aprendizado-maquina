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

