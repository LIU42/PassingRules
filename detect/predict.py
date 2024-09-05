import onnxruntime as ort
import utils.porcess as process

from data import TrafficSignal


class SignalDetector:
    def __init__(self, configs):
        if configs['device'] == 'CUDA':
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.configs = configs
        self.session = ort.InferenceSession(f'detect/weights/deploy/detect-{self.precision}.onnx', providers=providers)

    def __call__(self, image):
        inputs = process.preprocess(image, size=640, padding_color=127, precision=self.precision)

        outputs = self.session.run(None, inputs)
        outputs = self.reshape(outputs)
        
        results = process.non_max_suppression(outputs, self.conf_threshold, self.iou_threshold)

        return [TrafficSignal.from_bbox(box, color_index) for box, color_index in results]
    
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
    def reshape(outputs):
        return outputs[0].squeeze().transpose()
    