import matplotlib.pyplot as plt

# Create grid
x1list = np.linspace(np.min(X[:,0])-2, np.max(X[:,0])+2, 50)
x2list = np.linspace(np.min(X[:,1])-2, np.max(X[:,1])+2, 50)
xx, yy = np.meshgrid(x1list, x2list)
grid = np.column_stack((xx.ravel(), yy.ravel()))

# Predict model output on grid
grid_pred = model.predict(grid)

# Plot contour map of model output on grid
plt.contourf(xx, yy, grid_pred.reshape(xx.shape), cmap=plt.cm.coolwarm, alpha=0.8)

# Plot actual data points on top of contour map
plt.scatter(X[Y==0][:,0], X[Y==0][:,1], color='blue', label='Real')
plt.scatter(X[Y==1][:,0], X[Y==1][:,1], color='red', label='Fake')

plt.xlabel('Skewness of Wavelet Transformed Image')
plt.ylabel('Entropy of Image')
plt.legend()
plt.show()
