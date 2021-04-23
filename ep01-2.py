# weight initialization

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
    axes.set_ylim([xmin,ymax])
    plt.title("Weight:   w0=%.2f    w1=%.2f    w2=%.2f" %(w[0],w[1],w[2]))
    plt.scatter(X[correct,0],X[correct,1],c=colors_o, marker='o');
    plt.scatter(X[misclassified,0],X[misclassified,1],c=colors_x, marker='x');

    
# Starting weight vector <---- change as you wish (as long as w0[2] != 0)
w0 = np.asarray([[-0.5], [1] , [1]])
   
plot_state(Xe,w0)
