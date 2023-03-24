import matplotlib.pyplot as plt
import numpy as np
from urllib.request import urlopen
url = ’http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication'
raw_data = urlopen(url)
dataset = np.loadtxt(raw_data, delimiter=",")
print(dataset.shape)

X=dataset[:,[1,3]]
Y=dataset[:,4]
print(X.shape)
print(Y.shape)

X_fake = X[Y==1]
X_real = X[Y==0]

plt.scatter(X_fake[:,0], X_fake[:,1], color='red', label='Fake')
plt.scatter(X_real[:,0], X_real[:,1], color='blue', label='Real')

plt.xlabel('Skewness of Wavelet Transformed Image')
plt.ylabel('Entropy of Wavelet Transformed Image')
plt.legend()
plt.show()
