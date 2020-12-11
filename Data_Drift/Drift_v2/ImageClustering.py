import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import Utils

name = 'aruba'
labels = list(Utils.pick_dataset(name).label.unique())
labels.sort()

# image_width = LogClustering.LogClustering.WIDTH
# image_height = LogClustering.LogClustering.HEIGHT

input_folder = f"../../output/{name}/Daily_Images/"
list_files = glob.glob(input_folder + '*.png')

dataset = []
for file in list_files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    image_height, image_width = img.shape
    dataset.append(img.flatten())

dataset = np.asarray(dataset, dtype=int)

# range_n_clusters = [2, 3, 4, 5, 6]
range_n_clusters = [2, 3, 4, 5, 6]
# for n_clusters in range_n_clusters:
#     cluster_labels = KMeans(n_clusters=n_clusters, random_state=10).fit_predict(dataset)
#     silhouette_avg = silhouette_score(dataset, cluster_labels)
#
#     fig, ax = plt.subplots(1,1)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)
#
#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(dataset, cluster_labels)
#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belonging to
#         # cluster i, and sort them
#         ith_cluster_silhouette_values = \
#             sample_silhouette_values[cluster_labels == i]
#         ith_cluster_silhouette_values.sort()
#
#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i
#
#         color = cm.nipy_spectral(float(i) / n_clusters)
#         ax.fill_betweenx(np.arange(y_lower, y_upper),
#                           0, ith_cluster_silhouette_values,
#                           facecolor=color, edgecolor=color, alpha=0.7)
#
#         # Label the silhouette plots with their cluster numbers at the middle
#         ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#
#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10  # 10 for the 0 samples
#     ax.set_title("The silhouette plot for the various clusters.")
#     ax.set_xlabel("The silhouette coefficient values")
#     ax.set_ylabel("Cluster label")
#
#     # The vertical line for average silhouette score of all the values
#     ax.axvline(x=silhouette_avg, color="red", linestyle="--")
#
#     ax.set_yticks([])  # Clear the yaxis labels / ticks
#     ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#
#     plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
#                   "with n_clusters = %d" % n_clusters),
#                  fontsize=14, fontweight='bold')

n_clusters = None
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(dataset)
silhouette_avg = silhouette_score(dataset, clusters)

print("For n_clusters =", n_clusters,
      "The average silhouette_score is :", silhouette_avg)

# for center in kmeans.cluster_centers_:
#     center[(center >= 0.5)] = 255
#     center[(center < 0.5)] = 0

print(kmeans.cluster_centers_.shape)

fig, ax = plt.subplots(1, n_clusters)

centers = kmeans.cluster_centers_.reshape(n_clusters, image_height, image_width)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])

    axi.imshow(center, interpolation='nearest', cmap='gray')
yticks = []
yticks_labels = []
for i in range(len(labels)):
    s = i * image_height / len(labels)
    e = (i + 1) * image_height / len(labels)
    yticks.append((s + e) / 2)
    yticks_labels.append(labels[i])

ax[0].set_yticks(yticks)
ax[0].set_yticklabels(yticks_labels)
plt.show()
print(labels)

clusters_indices = []
for n in range(n_clusters):
    indices = [i for i, e in enumerate(clusters) if e == n]
