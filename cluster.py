import math

from entities import TrafficLight

class TrafficLightCluster:

    def __call__(self, detected_list: list[TrafficLight]) -> set[TrafficLight]:
        return self.cluster(detected_list)
    
    def create_set(self, *traffic_lights: TrafficLight) -> set[TrafficLight]:
        return set(traffic_lights)

    def get_closest_indices(self, cluster_list: list[set[TrafficLight]]) -> tuple[int, int, float]:
        min_similarity = math.inf
        closest_index1 = 0
        closest_index2 = 0

        for index1 in range(0, len(cluster_list) - 1):
            for index2 in range(index1 + 1, len(cluster_list)):
                similarity = TrafficLight.similarity(cluster_list[index1], cluster_list[index2])
                if similarity >= min_similarity:
                    continue
                min_similarity = similarity
                closest_index1 = index1
                closest_index2 = index2

        return closest_index1, closest_index2, min_similarity
    
    def get_closest_cluster(self, cluster_list: list[set[TrafficLight]]) -> set[TrafficLight]:
        if len(cluster_list) == 0:
            return set()
        closest_distance = math.inf
        closest_index = 0

        for index, cluster in enumerate(cluster_list, start = 0):
            distance = TrafficLight.center_distance(cluster)
            if distance >= closest_distance:
                continue
            closest_distance = distance
            closest_index = index

        return cluster_list[closest_index]

    def cluster(self, traffic_lights: list[TrafficLight], similarity_threshold: float = 30) -> set[TrafficLight]:
        cluster_list = list()
        for traffic_light in traffic_lights:
            cluster_list.append(self.create_set(traffic_light))

        while len(cluster_list) > 1:
            index1, index2, similarity = self.get_closest_indices(cluster_list)
            if similarity > similarity_threshold:
                break
            cluster_list[index1] = cluster_list[index1].union(cluster_list[index2])
            cluster_list.pop(index2)

        return self.get_closest_cluster(cluster_list)
