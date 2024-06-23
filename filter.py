import math
import statistics


class SignalFilter:

    def __init__(self, weights=(0.05, 5, 2), threshold=40):
        self.weight_x = weights[0]
        self.weight_y = weights[1]
        self.weight_size = weights[2]
        self.threshold = threshold

    def __call__(self, detections):
        return self.filter(detections)

    def similarity_to(self, signal1, signal2):
        distance_x = (signal1.center_x - signal2.center_x) * self.weight_x
        distance_y = (signal1.center_y - signal2.center_y) * self.weight_y

        size_similarity = (abs(signal1.width - signal2.width) + abs(signal1.height - signal2.height)) * self.weight_size
        return math.sqrt(distance_x ** 2 + distance_y ** 2) + size_similarity

    @staticmethod
    def create_set(*signals):
        return set(signals)

    @staticmethod
    def center_distance(signals):
        average_x = statistics.mean(signal.center_x for signal in signals)
        average_y = statistics.mean(signal.center_y for signal in signals)

        return math.sqrt((average_x - 320) ** 2 + (average_y - 240) ** 2)

    def similarity(self, cluster1, cluster2):
        min_distance = math.inf
        for signal1 in cluster1:
            for signal2 in cluster2:
                min_distance = min(self.similarity_to(signal1, signal2), min_distance)
        return min_distance

    def get_closest_indices(self, cluster_list):
        min_similarity = math.inf
        closest_index1 = 0
        closest_index2 = 0

        for index1 in range(0, len(cluster_list) - 1):
            for index2 in range(index1 + 1, len(cluster_list)):
                similarity = self.similarity(cluster_list[index1], cluster_list[index2])
                if similarity >= min_similarity:
                    continue
                min_similarity = similarity
                closest_index1 = index1
                closest_index2 = index2

        return closest_index1, closest_index2, min_similarity

    def get_closest_cluster(self, cluster_list):
        if len(cluster_list) == 0:
            return set()
        closest_distance = math.inf
        closest_index = 0

        for index, cluster in enumerate(cluster_list, start=0):
            distance = self.center_distance(cluster)
            if distance >= closest_distance:
                continue
            closest_distance = distance
            closest_index = index

        return cluster_list[closest_index]

    def filter(self, detections):
        cluster_list = list(map(self.create_set, detections))

        while len(cluster_list) > 1:
            index1, index2, similarity = self.get_closest_indices(cluster_list)
            if similarity > self.threshold:
                break
            cluster_list[index1] = cluster_list[index1].union(cluster_list[index2])
            cluster_list.pop(index2)

        return self.get_closest_cluster(cluster_list)
