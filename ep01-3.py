# exercise

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
    raise NotImplementedError("Function perceptron() is not implemented")
    # END YOUR CODE
    
    return w


# Test your function for w0, X and y as defined above
w0 = np.asarray([[-0.5], [1] , [1]]) # <---- you can change it (as long as w0[2] != 0)
print("Initial weight vector=\n", w0)

w = perceptron(Xe,y,w0,plot=True)


# Print the final weight vector and plot the final graph using the function plot_state.
# START OF YOUR CODE:
# END YOUR CODE
