import onnxruntime as ort

from wrappers import SignalBuilder
from utils import ImageUtils
from utils import ResultUtils


class SignalDetector:

    def __init__(self, device, precision, conf_threshold, iou_threshold):
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if device == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(f'detector/weights/product/detect-{precision}.onnx', providers=providers)

    def __call__(self, image):
        inputs = ImageUtils.preprocess(image, size=640, padding_color=127, precision=self.precision)

        outputs = self.session.run(None, {
            'images': inputs,
        })
        outputs = outputs[0].squeeze()
        outputs = outputs.transpose()
        
        results = ResultUtils.non_max_suppression(outputs, self.conf_threshold, self.iou_threshold)

        return [SignalBuilder.box(box, color_index) for box, color_index in results]
