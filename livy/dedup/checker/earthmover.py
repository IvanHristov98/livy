import livy.dedup.service as service

import numpy as np
import sklearn.cluster as cluster


class EarthMoverChecker(service.DuplicateChecker):
    _N_CLUSTERS = 10

    def is_duplicate(self, base_desc: np.ndarray, dup_desc: np.ndarray) -> bool:
        base_hist = self._hist(base_desc)
        dup_hist = self._hist(dup_desc)

        return False

    def _hist(self, desc: np.ndarray) -> np.ndarray:
        kmeans = cluster.KMeans(n_clusters=self._N_CLUSTERS, init="k-means++", random_state=0, n_init="auto").fit(desc)
        hist = np.zeros((self._N_CLUSTERS))
        
        for label in kmeans.labels_:
            hist[label] += 1

        return hist
