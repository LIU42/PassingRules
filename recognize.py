from classify import ShapeClassifier
from data import PassingDirects
from detect import SignalDetector
from filter import SignalFilter


class RulesRecognizer:
    def __init__(self, configs):
        self.configs = configs
        self.detector = SignalDetector(configs)
        self.filter = SignalFilter(configs)
        self.classifier = ShapeClassifier(configs)

        if self.strategy == 'radical':
            self.is_passable = lambda signal: signal.color != 'red'
        else:
            self.is_passable = lambda signal: signal.color == 'green'

    def __call__(self, image):
        detections = self.detector(image)

        if len(detections) == 0:
            return [], PassingDirects.allow()

        signals = self.filter(detections)
        signals = self.classifier(image, signals)

        return signals, self.recognize(signals)
    
    @property
    def strategy(self):
        return self.configs['strategy']

    def global_recognize(self, signals):
        if self.strategy == 'radical':
            if any(signal.shape == 'full' and not self.is_passable(signal) for signal in signals):
                return PassingDirects.prohibit()
            else:
                return PassingDirects.allow()
        else:
            if any(signal.shape == 'full' and self.is_passable(signal) for signal in signals):
                return PassingDirects.allow()
            else:
                return PassingDirects.prohibit()

    def recognize(self, signals):
        passing_directs = self.global_recognize(signals)

        for signal in filter(lambda signal: signal.shape != 'full', signals):
            if signal.shape == 'straight':
                passing_directs.straight = self.is_passable(signal)

            elif signal.shape == 'left':
                passing_directs.left = self.is_passable(signal)

            elif signal.shape == 'right':
                passing_directs.right = self.is_passable(signal)

        return passing_directs
