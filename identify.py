from classifier.predict import ShapeClassifier
from detector.predict import TrafficLightDetector

from filter import TrafficLightFilter
from utils import PlottingUtils

from structures import TrafficLight
from structures import TrafficSignal


class TrafficSignalIdentifier:

    def __init__(self):
        self.detector = TrafficLightDetector()
        self.filter = TrafficLightFilter()
        self.classifier = ShapeClassifier()

    def __call__(self, image, strategy="conservative", plot_result=True):
        return self.identify(image, strategy, plot_result)

    def detect_traffic_lights(self, image):
        return self.classifier(image, self.filter(self.detector(image)))

    @staticmethod
    def get_passable_judge(strategy):
        if strategy == "conservative":
            return lambda color: color == "green"
        if strategy == "radical":
            return lambda color: color != "red"

    def identify_global_signal(self, lights, signal):
        global_forbid = False
        global_allow = False
        is_passable = self.get_passable_judge(signal.strategy)

        for light in lights:
            if light.shape != "full":
                continue
            if is_passable(light.color):
                global_allow = True
            else:
                global_forbid = True

        if signal.strategy == "conservative" and global_allow and not global_forbid:
            signal.allow_all()
        elif signal.straight == "radical" and global_forbid and not global_allow:
            signal.forbid_all()

        return signal

    def identify_direct_signal(self, lights, signal):
        is_passable = self.get_passable_judge(signal.strategy)

        for light in lights:
            if light.shape == "full":
                continue
            elif light.shape == "straight":
                signal.straight = is_passable(light.color)
            elif light.shape == "left":
                signal.left = is_passable(light.color)
            elif light.shape == "right":
                signal.right = is_passable(light.color)

        return signal

    def identify(self, image, strategy, plot_result):
        signal = TrafficSignal(strategy)
        lights = self.detect_traffic_lights(image)

        if len(lights) == 0:
            signal.allow_all()
            return signal

        signal = self.identify_global_signal(lights, signal)
        signal = self.identify_direct_signal(lights, signal)

        if plot_result:
            PlottingUtils.plot_traffic_lights(image, lights)
            PlottingUtils.plot_traffic_signal(image, signal)

        return signal
