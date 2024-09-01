import onnxruntime as ort

import utils.preporcess as preprocess
import utils.postprocess as postprocess

from wrappers import TrafficSignal


class SignalDetector:
    def __init__(self, configs):
        if configs['device'] == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.configs = configs
        self.session = ort.InferenceSession(f'detect/weights/product/detect-{self.precision}.onnx', providers=providers)

    def __call__(self, image):
        inputs = preprocess.preprocess(image, size=640, padding_color=127, precision=self.precision)

        outputs = self.session.run([], inputs)
        outputs = self.postprocessing(outputs)
        
        results = postprocess.non_max_suppression(outputs, self.conf_threshold, self.iou_threshold)

        return [TrafficSignal.from_box(box, color_index) for box, color_index in results]
    
    @property
    def precision(self):
        return self.configs['precision']
    
    @property
    def conf_threshold(self):
        return self.configs['detect']['conf-threshold']

    @property
    def iou_threshold(self):
        return self.configs['detect']['iou-threshold']

    @staticmethod
    def postprocessing(outputs):
        return outputs[0].squeeze().transpose()
    