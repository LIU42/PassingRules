from classifier.predict import ShapeClassifier
from detector.predict import SignalDetector

from filter import SignalFilter
from structs import PassingRules
from utils import PlottingUtils


class RulesRecognizer:

    def __init__(self, **arguments):
        self.strategy = arguments['strategy']
        self.plotting = arguments['plotting']

        self.detector = SignalDetector(
            conf_threshold=arguments['conf_threshold'],
            iou_threshold=arguments['iou_threshold'],
            device=arguments['device'],
            precision=arguments['precision'],
        )
        self.filter = SignalFilter(
            weights=arguments['filter_weights'],
            threshold=arguments['filter_threshold'],
        )
        self.classifier = ShapeClassifier(
            device=arguments['device'],
            precision=arguments['precision'],
        )
        self.is_passable = self.get_passable_judge()

    def __call__(self, image):
        return self.recognize(image)

    def detect_traffic_signals(self, image):
        return self.classifier(image, self.filter(self.detector(image)))
    
    def get_passable_judge(self):
        if self.strategy == 'conservative':
            return lambda color: color == 'green'
        else:
            return lambda color: color != 'red'
        
    def create_passing_rules(self):
        return PassingRules(self.strategy)

    def recognize_global_rules(self, signals, rules):
        global_allow = False
        global_forbid = False

        for signal in signals:
            if signal.shape != 'full':
                continue
            if self.is_passable(signal.color):
                global_allow = True
            else:
                global_forbid = True

        if self.strategy == 'conservative' and global_allow and not global_forbid:
            rules.allow_all()
        elif self.strategy == 'radical' and global_forbid and not global_allow:
            rules.forbid_all()

        return rules

    def recognize_direct_rules(self, signals, rules):
        for signal in signals:
            if signal.shape == 'full':
                continue
            elif signal.shape == 'straight':
                rules.straight = self.is_passable(signal.color)
            elif signal.shape == 'left':
                rules.left = self.is_passable(signal.color)
            elif signal.shape == 'right':
                rules.right = self.is_passable(signal.color)
        return rules

    def recognize(self, image):
        signals = self.detect_traffic_signals(image)
        rules = self.create_passing_rules()

        if len(signals) > 0:
            rules = self.recognize_global_rules(signals, rules)
            rules = self.recognize_direct_rules(signals, rules)

            if self.plotting:
                PlottingUtils.plot_traffic_signals(image, signals)
                PlottingUtils.plot_passing_rules(image, rules)
        else:
            rules.allow_all()

        return rules
