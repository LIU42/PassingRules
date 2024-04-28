import cv2

from entities import TrafficSignal
from entities import TrafficLight

from detect.predict import TrafficLightDetector
from cluster import TrafficLightCluster
from classify.predict import ShapeClassifier

class TrafficSignalIdentifier:

    def __init__(self) -> None:
        self.detector = TrafficLightDetector()
        self.cluster = TrafficLightCluster()
        self.classifier = ShapeClassifier()

    def __call__(self, image: cv2.Mat, strategy: str = "conservative", plot_result: bool = False) -> TrafficSignal:
        assert strategy == "radical" or strategy == "conservative"
        if strategy == "radical":
            return self.radical_identify(image, plot_result)
        elif strategy == "conservative":
            return self.conservative_identify(image, plot_result)

    def get_traffic_lights(self, image: cv2.Mat, plot_result: bool = False) -> set[TrafficLight]:
        traffic_lights = self.classifier(image, self.cluster(self.detector(image)))
        if plot_result:
            for traffic_light in traffic_lights:
                image = traffic_light.plot(image)
        return traffic_lights
    
    def identify_full_signal(self, traffic_light_group: set[TrafficLight]) -> tuple[bool, bool]:
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
    
    def identify_direct_signal(self, traffic_light_group: set[TrafficLight], traffic_signal: TrafficSignal) -> TrafficSignal:
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
        traffic_lights = self.get_traffic_lights(image, plot_result)
        for traffic_light in traffic_lights:
            traffic_light.set_conservative()

        traffic_signal = TrafficSignal(strategy = "conservative")
        if len(traffic_lights) == 0:
            traffic_signal.all_allow()
            return traffic_signal
        
        have_green_full, have_red_full = self.identify_full_signal(traffic_lights)
        if have_green_full and not have_red_full:
            traffic_signal.all_allow()

        traffic_signal = self.identify_direct_signal(traffic_lights, traffic_signal)
        if plot_result:
            image = traffic_signal.plot(image)

        return traffic_signal

    def radical_identify(self, image: cv2.Mat, plot_result: bool = False) -> TrafficSignal:
        traffic_lights = self.get_traffic_lights(image, plot_result)
        for traffic_light in traffic_lights:
            traffic_light.set_radical()

        traffic_signal = TrafficSignal(strategy = "radical")
        if len(traffic_lights) == 0:
            traffic_signal.all_allow()
            return traffic_signal
        
        have_green_full, have_red_full = self.identify_full_signal(traffic_lights)
        if have_red_full and not have_green_full:
            traffic_signal.all_forbid()
            
        traffic_signal = self.identify_direct_signal(traffic_lights, traffic_signal)
        if plot_result:
            image = traffic_signal.plot(image)

        return traffic_signal
