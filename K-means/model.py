import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    def __init__(self, K=3, max_iters=100, plot_steps=False) -> None:
        self.X = None
        self.n_sample = None
        self.K = K
        self.max_iters = max_iters
        self.centroids = []
        self.clusters = [[] for _ in range(self.K)]
        self.plot_steps = plot_steps

    def _get_cluster_labels(self, clusters) -> np.array:
        labels = np.empty(self.n_sample)
        for cluster_idx, cluster in clusters:
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _get_closest_centroid(self, x, centroids) -> int:
        distances = [euclidean_distance(x, centroid) for centroid in centroids]
        return np.argmin(distances)

    def _create_clusters(self, centroids) -> np.array:
        clusters = [[] for _ in range(self.K)]
        for idx, x in enumerate(self.X):
            label = self._get_closest_centroid(x, centroids)
            clusters[label].append(idx)
        return clusters

    def _get_centroids(self, clusters) -> np.array:
        centroids = np.zeros((self.K, self.n_feature))
        for label, cluster in enumerate(clusters):
            cluster_means = np.mean(self.X[cluster], axis=0)
            centroids[label] = cluster_means

        return centroids

    def _is_converged(self, old_centroids, new_centroids):
        distances = [
            euclidean_distance(old_centroids[i], new_centroids[i])
            for i in range(self.K)
        ]
        return sum(distances) == 0

    def plot(self) -> None:
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

    def predict(self, X) -> np.array:
        # Init X, , n_sample, centroids
        self.X = X
        self.n_sample, self.n_feature = X.shape

        random_idxs = np.random.choice(self.n_sample, self.K, replace=False)
        self.centroids = [X[idx] for idx in random_idxs]

        # Iterate this actions
        for _ in range(self.max_iters):
            #  Get clusters by centroids
            # : Update centroids
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()

            old_centroids = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(old_centroids, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        return self._get_cluster_labels(self.clusters)
