from classifier.predict import MainClassifier
from detector.predict import MainDetector

from filter import MainFilter
from structs import TrafficSignal
from utils import PlottingUtils


class MainPredictor:

    def __init__(self, **arguments):
        self.strategy = arguments['strategy']
        self.plotting = arguments['plotting']

        self.detector = MainDetector(
            conf_threshold=arguments['conf_threshold'],
            nms_threshold=arguments['nms_threshold'],
        )
        self.filter = MainFilter(
            threshold=arguments['filter_threshold'],
            weights=arguments['filter_weights'],
        )
        self.classifier = MainClassifier()
        self.is_passable = self.get_passable_judge()

    def __call__(self, image):
        return self.predict(image)

    def create_traffic_signal(self):
        return TrafficSignal(self.strategy)

    def detect_traffic_lights(self, image):
        return self.classifier(image, self.filter(self.detector(image)))

    def get_passable_judge(self):
        if self.strategy == 'conservative':
            return lambda color: color == 'green'
        if self.strategy == 'radical':
            return lambda color: color != 'red'

    def predict_global_signal(self, lights, signal):
        global_forbid = False
        global_allow = False

        for light in lights:
            if light.shape != 'full':
                continue
            if self.is_passable(light.color):
                global_allow = True
            else:
                global_forbid = True

        if self.strategy == 'conservative' and global_allow and not global_forbid:
            signal.allow_all()
        elif self.strategy == 'radical' and global_forbid and not global_allow:
            signal.forbid_all()

        return signal

    def predict_direct_signal(self, lights, signal):
        for light in lights:
            if light.shape == 'full':
                continue
            elif light.shape == 'straight':
                signal.straight = self.is_passable(light.color)
            elif light.shape == 'left':
                signal.left = self.is_passable(light.color)
            elif light.shape == 'right':
                signal.right = self.is_passable(light.color)
        return signal

    def predict(self, image):
        signal = self.create_traffic_signal()
        lights = self.detect_traffic_lights(image)

        if len(lights) == 0:
            signal.allow_all()
            return signal

        signal = self.predict_global_signal(lights, signal)
        signal = self.predict_direct_signal(lights, signal)

        if self.plotting:
            PlottingUtils.plot_traffic_lights(image, lights)
            PlottingUtils.plot_traffic_signal(image, signal)

        return signal
