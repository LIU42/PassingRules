import math
import statistics

from structures import TrafficLight


class TrafficLightFilter:

    def __init__(self, center=(320, 240), weight=(0.05, 5, 2)):
        self.center_x = center[0]
        self.center_y = center[1]
        self.weight_x = weight[0]
        self.weight_y = weight[1]
        self.weight_size = weight[2]

    def __call__(self, detections, threshold=40):
        return self.do_filter(detections, threshold)

    def similarity_to(self, light1, light2):
        distance_x = (light1.center_x - light2.center_x) * self.weight_x
        distance_y = (light1.center_y - light2.center_y) * self.weight_y

        size_similarity = (abs(light1.width - light2.width) + abs(light1.height - light2.height)) * self.weight_size
        return math.sqrt(distance_x ** 2 + distance_y ** 2) + size_similarity

    @staticmethod
    def create_set(*lights):
        return set(lights)

    def center_distance(self, lights):
        average_x = statistics.mean(light.center_x for light in lights)
        average_y = statistics.mean(light.center_y for light in lights)

        return math.sqrt((average_x - self.center_x) ** 2 + (average_y - self.center_y) ** 2)

    def similarity(self, cluster1, cluster2):
        min_distance = math.inf
        for light1 in cluster1:
            for light2 in cluster2:
                min_distance = min(self.similarity_to(light1, light2), min_distance)
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

    def do_filter(self, detected_list, threshold):
        cluster_list = list(map(self.create_set, detected_list))

        while len(cluster_list) > 1:
            index1, index2, similarity = self.get_closest_indices(cluster_list)
            if similarity > threshold:
                break
            cluster_list[index1] = cluster_list[index1].union(cluster_list[index2])
            cluster_list.pop(index2)

        return self.get_closest_cluster(cluster_list)
