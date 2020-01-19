import sys
import numpy as np
import scipy.io.wavfile as wav

sample, centroids = sys.argv[1], sys.argv[2]
fs, y = wav.read(sample)
x = np.array(y.copy())
centroids = np.loadtxt(centroids)
k = len(centroids)
n = len(y)
last_centroids = np.zeros(centroids.shape)
new_centroids = centroids
clusters = np.zeros(k)
distances = np.zeros((n, k))
error = np.linalg.norm(new_centroids - last_centroids)
output = open("output2.txt", "w")

for i in range(30):
    if error == 0:
        break
    # Calculate the distances between each data point to the centroids
    for row in range(n):
        for c in range(k):
            distances[row][c] = np.linalg.norm(y[row] - new_centroids[c])
    # Choose the minimum index of each row to assign the clusters
    clusters = np.argmin(distances, axis=1)
    # deep copy the last centroids
    last_centroids = np.copy(new_centroids)
    # calculate the mean of each cluster by using the indices of the clusters to find the points
    # round the output
    for j in range(k):
        new_centroids[j] = np.round(np.mean(y[clusters == j], axis=0))
        # union all the old data to the centroid to create the compressed file
        x[clusters == j] = new_centroids[j]
    # calc the error different
    error = np.linalg.norm(new_centroids - last_centroids)
    print(f"[iter {i}]:{','.join([str(i) for i in new_centroids])}")
    output.write(f"[iter {i}]:{','.join([str(i) for i in new_centroids])}\n")
wav.write("compressed.wav", fs, np.array(x, dtype=np.int16))
output.close()
