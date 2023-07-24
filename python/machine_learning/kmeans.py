import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from abc import abstractmethod


def plot(X, centroids, labels, show=True, iteration=None, file_name=None):
    # Plot the original data and clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)
    plt.title('K-means Clustering iteration ' + str(iteration))
    plt.xlabel('X')
    plt.ylabel('Y')
    if show:
        plt.show()
    if iteration and file_name:
        file_name = file_name + "_" + str(iteration)
        plt.savefig(file_name)
    plt.close()

def comparison_plot(X, c1, l1, c2, l2):
    # Plot the original data and clusters
    plt.scatter(X[:, 0], X[:, 1], c=l1, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=l2, cmap='viridis')
    plt.scatter(c1[:, 0], c1[:, 1], c='red', marker='x', s=100)
    plt.scatter(c2[:, 0], c2[:, 1], c='orange', marker='*', s=100)
    plt.title('K-means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


class BaseModel(ABC):
    
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    
class KMeans(BaseModel):
    
    def __init__(self, n_clusters, max_iter, file_prefix=None) -> None:
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._centroids = None
        self._labels = None
        self._file_prefix = file_prefix
        self._init_centroids = None
        
    @property
    def lables(self):
        return self._labels
    
    @property
    def centroids(self):
        return self._centroids
    
    @property
    def initial_centroids(self):
        return self._init_centroids
    
    def _initialize_centroids(self, K, X):
        centroid_indices = np.random.choice(len(X), K, replace=False)
        centroids = X[centroid_indices.tolist()]
        self._init_centroids = centroids
        return centroids
    
    def _calculate_euclidean_distance(self, centroids, X):
        # return distance per each centroid
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        return distances
    
    def _assign_labels(self, distances):
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, n_clusters, labels):
        new_centroids = []
        for i in range(n_clusters):
            new_centroids.append(np.mean(X[labels==i], axis=0)) # take average row wise (or per data point)
        return np.array(new_centroids)    
        
    def fit(self, X, y=None):
        centroids = self._initialize_centroids(self._n_clusters, X)
        labels = None
        for i in range(self._max_iter):
            distances = self._calculate_euclidean_distance(centroids, X)

            labels = self._assign_labels(distances)
        
            centroids = self._update_centroids(X, self._n_clusters, labels)
            
            if self._file_prefix:
                plot(X, centroids, labels, False, i, self._file_prefix)

        self._centroids = centroids
        self._labels = labels
    
    def predict(self, X):
        return NotImplemented("Not implemented")



np.random.seed(123)

K = 3
N = 10000
M = 2

X = np.random.rand(N, M)
max_iter = 30

# plot results per iteration and visualize 
# kmeans = KMeans(n_clusters=K, max_iter=max_iter, file_prefix="kmeans_plots/kmeans_clustering")

# no visualization
kmeans = KMeans(n_clusters=K, max_iter=max_iter)
kmeans.fit(X)

from sklearn.cluster import KMeans as SciktKMeans

scikit_kmeans = SciktKMeans(init=kmeans.initial_centroids, n_clusters=K, random_state=123, max_iter=max_iter).fit(X)
print(scikit_kmeans.cluster_centers_)
print(kmeans.centroids)

plot(X, kmeans.centroids, kmeans.lables)
plot(X, scikit_kmeans.cluster_centers_, scikit_kmeans.labels_)

comparison_plot(X, scikit_kmeans.cluster_centers_, scikit_kmeans.labels_, kmeans.centroids, kmeans.lables)

## generate gif
#from util import generate_gif

#input_path = "kmeans_plots"
#file_prefix = "kmeans_clustering"
#generate_gif(path=input_path, file_prefix=file_prefix, output_path="images", output_file="kmeans_clustering_animate.gif", duration=1000)
