import collections
import math
import numpy as np
import statistics

from sklearn.cluster import DBSCAN


class SignalFilter:
    def __init__(self, configs):
        self.configs = configs
        self.cluster = DBSCAN(eps=self.threshold, min_samples=1, metric='precomputed', n_jobs=-1)

    def __call__(self, detections):
        clusters = collections.defaultdict(list)

        for index, label in enumerate(self.cluster.fit(self.difference_matrix(detections)).labels_):
            if label >= 0:
                clusters[label].append(detections[index])

        return self.result_cluster(clusters)

    @property
    def weight_x(self):
        return self.configs['filter']['weights'][0]

    @property
    def weight_y(self):
        return self.configs['filter']['weights'][1]

    @property
    def weight_w(self):
        return self.configs['filter']['weights'][2]

    @property
    def weight_h(self):
        return self.configs['filter']['weights'][3]
    
    @property
    def threshold(self):
        return self.configs['filter']['threshold']

    def rect_difference(self, rect1, rect2):
        difference_x = (rect1.center_x - rect2.center_x) * self.weight_x
        difference_y = (rect1.center_y - rect2.center_y) * self.weight_y

        difference_w = abs(rect1.w - rect2.w) * self.weight_w
        difference_h = abs(rect1.h - rect2.h) * self.weight_h

        return math.sqrt(difference_x ** 2 + difference_y ** 2) + difference_w + difference_h

    def difference_matrix(self, rects):
        size = len(rects)
        differences = np.zeros((size, size), dtype=np.float32)

        for i in range(size):
            for j in range(size):
                differences[i, j] = self.rect_difference(rects[i], rects[j])

        return differences

    @staticmethod
    def result_cluster(clusters):
        distances = np.zeros(len(clusters), dtype=np.float32)

        for label, cluster in clusters.items():
            average_x = statistics.mean(rect.center_x for rect in cluster)
            average_y = statistics.mean(rect.center_y for rect in cluster)

            distances[label] = math.sqrt((average_x - 320) ** 2 + (average_y - 240) ** 2)

        return clusters[np.argmin(distances)]
