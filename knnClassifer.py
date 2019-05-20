from builtins import range
from builtins import object
import numpy as np

class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X, k=1, num_loops=0):
        # compute distances between data to be tested(X) and the train data(self.X)
        dists = self.compute_distances(X)

        # return the most common label among the chosen k ones
        return self.predict_labels(dists, k=k)

    def compute_distances(self, X):
        sum_X_sq = np.sum(X**2, axis=1)[:, np.newaxis] # x^2
        sum_X_train_sq = np.sum(self.X_train**2, axis=1) # y^2
        mul_X_X_train = -2*np.dot(X, self.X_train.T) # -2xy

        # dist = sqrt((x-y)^2) = sqrt(x^2 + y^2 - 2xy)
        # ^can remove sqrt, wont make a diff

        dists = sum_X_sq + sum_X_train_sq + mul_X_X_train

        return dists

    def predict_labels(self, dists, k=1):
        # get k smallest values from dists and return mode of the labels.
        all_idxs = np.argsort(dists, axis=1)
        y_pred = np.zeros(dists.shape[0])
        for i, idx in enumerate(all_idxs): # k smallest dists
            closest_y = self.y_train[idx[:k]]
            counts = np.bincount(closest_y)
            y_pred[i] = np.argmax(counts)

        return y_pred # predicted labels for each test data point