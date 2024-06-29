import onnxruntime as ort

from structs import TrafficSignal
from utils import ImageUtils
from utils import ResultUtils


class SignalDetector:

    def __init__(self, device='CPU', precision='fp32', conf_threshold=0.25, iou_threshold=0.45):
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if device == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(f'detector/weights/signal-detect-{precision}.onnx', providers=providers)

    def __call__(self, image):
        return self.detect(image)

    @staticmethod
    def get_color(color_index):
        if color_index == 0:
            return 'red'
        if color_index == 1:
            return 'green'
        if color_index == 2:
            return 'yellow'
        return None

    def detect(self, image):
        inputs = ImageUtils.preprocess(image, size=640, padding_color=127, precision=self.precision)

        outputs = self.session.run(None, {
            'images': inputs,
        })
        outputs = outputs[0].squeeze()
        outputs = outputs.transpose()

        results = ResultUtils.non_max_suppression(outputs, self.conf_threshold, self.iou_threshold)
        detections = list()

        for (x, y, width, height), color_index in results:
            detections.append(TrafficSignal(x, y - 80, width, height, self.get_color(color_index)))

        return detections
