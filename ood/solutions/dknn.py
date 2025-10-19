class DKNN():
    def __init__(self, k=5):
        self.k = k
        self.fit_features = None

    def fit(self, features):
        self.fit_features = features

    def _compute_distances(self, test_feature):
        distances = np.linalg.norm(self.fit_features - test_feature, axis=1)
        return distances

    def _distance_to_kth_nn(self, distances):
        sorted_distances = np.sort(distances)
        return sorted_distances[self.k - 1]

    def compute_scores(self, test_features):
        scores = []
        for test_feature in test_features:
            distances = self._compute_distances(test_feature)
            score = self._distance_to_kth_nn(distances)
            scores.append(score)
        return np.array(scores)