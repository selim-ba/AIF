class Mahalanobis():
    def __init__(self):
        self.mus = None
        self.inv_cov = None
        self.labels = None

    def fit(self, features, labels):
        self.labels = np.unique(labels)
        self.mus = {}
        covs = {}
        for label in self.labels:
            label_features = features[labels == label]
            self.mus[label] = np.mean(label_features, axis=0)
            covs[label] = np.cov(label_features.T)

        cov = np.sum(list(covs.values()), axis=0) / len(features)
        self.inv_cov = np.linalg.pinv(cov)

    def _mahalanobis_distance(self, x, mu, inv_cov):
        diff = x - mu
        return np.dot(np.dot(diff, inv_cov), diff)

    def compute_scores(self, test_features):
        scores = []
        for test_feature in test_features:
            distances = [self._mahalanobis_distance(test_feature, self.mus[label], self.inv_cov) for label in self.labels]
            scores.append(- np.min(distances))
        return scores