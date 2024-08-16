import onnxruntime as ort

from wrappers import SignalBuilder
from utils import ImageUtils
from utils import ResultUtils


class SignalDetector:
    def __init__(self, configs):
        if configs['device'] == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.configs = configs
        self.session = ort.InferenceSession(f'detector/weights/product/detect-{self.precision}.onnx', providers=providers)

    def __call__(self, image):
        inputs = ImageUtils.preprocess(image, size=640, padding_color=127, precision=self.precision)

        outputs = self.session.run([], inputs)
        outputs = self.postprocess(outputs)
        
        results = ResultUtils.non_max_suppression(outputs, self.conf_threshold, self.iou_threshold)

        return [SignalBuilder.box(box, color_index) for box, color_index in results]
    
    @property
    def precision(self):
        return self.configs['precision']
    
    @property
    def conf_threshold(self):
        return self.configs['detector']['conf-threshold']

    @property
    def iou_threshold(self):
        return self.configs['detector']['iou-threshold']

    @staticmethod
    def postprocess(outputs):
        return outputs[0].squeeze().transpose()
    