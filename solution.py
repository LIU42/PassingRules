import cv2

from entity import TrafficSignal
from entity import TrafficLight
from cluster import TrafficLightCluster
from detect.predict import TrafficLightDetector
from classify.predict import ShapeClassifier

class TrafficSignalSolution:

    def __init__(self) -> None:
        self.detector = TrafficLightDetector()
        self.cluster = TrafficLightCluster()
        self.classifier = ShapeClassifier()

    def __call__(self, image: cv2.Mat, strategy: str = "conservative", plot_result: bool = False) -> TrafficSignal:
        if strategy == "radical":
            return self.radical_identify(image, plot_result)
        return self.conservative_identify(image, plot_result)

    def get_traffic_lights(self, image: cv2.Mat, plot_result: bool = False) -> set[TrafficLight]:
        traffic_light_group = self.classifier(image, self.cluster(self.detector(image)))
        if plot_result:
            for traffic_light in traffic_light_group:
                image = traffic_light.plot(image)
        return traffic_light_group
    
    def parse_full_signal(self, traffic_light_group: set[TrafficLight]) -> tuple[bool, bool]:
        have_green_full = False
        have_red_full = False
        for traffic_light in traffic_light_group:
            if traffic_light.shape != "full":
                continue
            elif traffic_light.color == "green":
                have_green_full = True
            elif traffic_light.color == "red":
                have_red_full = True
        return have_green_full, have_red_full
    
    def parse_direct_signal(self, traffic_light_group: set[TrafficLight], traffic_signal: TrafficSignal) -> TrafficSignal:
        for traffic_light in traffic_light_group:
            if traffic_light.shape == "full":
                continue
            elif traffic_light.shape == "straight":
                traffic_signal.straight = (traffic_light.color == "green")
            elif traffic_light.shape == "left":
                traffic_signal.left = (traffic_light.color == "green")
            elif traffic_light.shape == "right":
                traffic_signal.right = (traffic_light.color == "green")
        return traffic_signal

    def conservative_identify(self, image: cv2.Mat, plot_result: bool = False) -> TrafficSignal:
        traffic_light_group = self.get_traffic_lights(image, plot_result)
        for traffic_light in traffic_light_group:
            if traffic_light.color != "green":
                traffic_light.color = "red"

        traffic_signal = TrafficSignal(strategy = "conservative")
        if len(traffic_light_group) == 0:
            traffic_signal.all_allow()
            return traffic_signal
        
        have_green_full, have_red_full = self.parse_full_signal(traffic_light_group)
        if have_green_full and not have_red_full:
            traffic_signal.all_allow()

        traffic_signal = self.parse_direct_signal(traffic_light_group, traffic_signal)
        if plot_result:
            image = traffic_signal.plot(image)
        return traffic_signal

    def radical_identify(self, image: cv2.Mat, plot_result: bool = False) -> TrafficSignal:
        traffic_light_group = self.get_traffic_lights(image, plot_result)
        for traffic_light in traffic_light_group:
            if traffic_light.color != "red":
                traffic_light.color = "green"

        traffic_signal = TrafficSignal(strategy = "radical")
        if len(traffic_light_group) == 0:
            traffic_signal.all_allow()
            return traffic_signal
        
        have_green_full, have_red_full = self.parse_full_signal(traffic_light_group)
        if have_red_full and not have_green_full:
            traffic_signal.all_forbid()
            
        traffic_signal = self.parse_direct_signal(traffic_light_group, traffic_signal)
        if plot_result:
            image = traffic_signal.plot(image)
        return traffic_signal
