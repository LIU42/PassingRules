import cv2

from classifier.predict import ShapeClassifier
from cluster import TrafficLightCluster
from detector.predict import TrafficLightDetector

from entities import TrafficLight
from entities import TrafficSignal

class TrafficSignalIdentifier:

    def __init__(self) -> None:
        self.detector = TrafficLightDetector()
        self.classifier = ShapeClassifier()
        self.cluster = TrafficLightCluster()

    def __call__(self, image: cv2.Mat, strategy: str = "conservative", plot_result: bool = True) -> TrafficSignal:
        return self.identify(image, strategy, plot_result)

    def get_traffic_lights(self, image: cv2.Mat, plot_result: bool = True) -> set[TrafficLight]:
        traffic_lights = self.classifier(image, self.cluster(self.detector(image)))
        if plot_result:
            image = TrafficLight.plot_all(image, traffic_lights)
        return traffic_lights
    
    def handle_global_signal(self, traffic_lights: set[TrafficLight], traffic_signal: TrafficSignal) -> TrafficSignal:
        have_global_red = False
        have_global_green = False

        for traffic_light in traffic_lights:
            if traffic_light.shape != "full":
                continue
            elif traffic_light.color == "red":
                have_global_red = True
            elif traffic_light.color == "green":
                have_global_green = True

        if traffic_signal.strategy == "conservative" and have_global_green and not have_global_red:
            traffic_signal.allow_all()
        elif traffic_signal.straight == "radical" and have_global_red and not have_global_green:
            traffic_signal.forbid_all()

        return traffic_signal
    
    def handle_direct_signal(self, traffic_lights: set[TrafficLight], traffic_signal: TrafficSignal) -> TrafficSignal:
        for traffic_light in traffic_lights:
            if traffic_light.shape == "full":
                continue
            elif traffic_light.shape == "straight":
                traffic_signal.straight = (traffic_light.color == "green")
            elif traffic_light.shape == "left":
                traffic_signal.left = (traffic_light.color == "green")
            elif traffic_light.shape == "right":
                traffic_signal.right = (traffic_light.color == "green")
        return traffic_signal
    
    def identify(self, image: cv2.Mat, strategy: str = "conservative", plot_result: bool = True) -> TrafficSignal:
        assert strategy == "conservative" or strategy == "radical"
        traffic_signal = TrafficSignal(strategy)
        traffic_lights = self.get_traffic_lights(image, plot_result)

        if len(traffic_lights) == 0:
            traffic_signal.allow_all()
            return traffic_signal
        
        for traffic_light in traffic_lights:
            if strategy == "conservative":
                traffic_light.set_conservative()
            elif strategy == "radical":
                traffic_light.set_radical()

        traffic_signal = self.handle_global_signal(traffic_lights, traffic_signal)
        traffic_signal = self.handle_direct_signal(traffic_lights, traffic_signal)

        if plot_result:
            image = traffic_signal.plot(image)

        return traffic_signal
