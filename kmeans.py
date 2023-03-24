import numpy as np

def update_assignments(data, centroids):
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(data, old_centroids, assignments):
    new_centroids = np.zeros_like(old_centroids)
    for i in range(old_centroids.shape[0]):
        new_centroids[i] = np.mean(data[assignments == i], axis=0)
    return new_centroids
  """""""""""""""""""""""""""""""""""""""""""""""""
  """""""""""""""""""""""""""""""""""""""""""""""""
  """""""""""""""""""""""""""""""""""""""""""""""""
  import csv

# Load the data from the CSV file
data = []
with open('blobs.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append([float(x) for x in row])
data = np.array(data)

# Initialize the centroids
np.random.seed(0)
initial_centroids = data[np.random.choice(data.shape[0], size=3, replace=False), :]

# Perform K-Means clustering
centroids = initial_centroids.copy()
for i in range(5):
    assignments = update_assignments(data, centroids)
    centroids = update_centroids(data, centroids, assignments)

# Print the final centroids and assignments
print('Final centroids:', centroids)
print('Final assignments:', assignments)
"""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""
# Plot the data and centroids
import matplotlib.pyplot as plt
plt.scatter(data[:, 0], data[:, 1], c=assignments)
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', marker='x', s=100)
plt.show()


losses = []
for k in range(1, 11):
    centroids = data[np.random.choice(data.shape[0], size=k, replace=False), :]
    for i in range(5):
        assignments = update_assignments(data, centroids)
        centroids = update_centroids(data, centroids, assignments)
    loss = ((data - centroids[assignments])**2).sum()
    losses.append(loss)

plt.plot(range(1, 11), losses, '-o')
plt.xlabel('Number of centroids (K)')
plt.ylabel('Loss')
plt.show()
