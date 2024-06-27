import numpy as np
import onnxruntime as ort

from utils import ImageUtils


class ShapeClassifier:

    def __init__(self, device='CPU', precision='fp32'):
        if device == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(f'./classifier/weights/shape-classify-{precision}.onnx', providers=providers)
        self.precision = precision

    def __call__(self, image, signals):
        return self.classify(image, signals)

    @staticmethod
    def get_shape(shape_index):
        if shape_index == 0:
            return 'full'
        if shape_index == 1:
            return 'left'
        if shape_index == 2:
            return 'right'
        if shape_index == 3:
            return 'straight'
        return None

    def classify(self, image, signals):
        for signal in signals:
            x1 = signal.x1
            y1 = signal.y1
            x2 = signal.x2
            y2 = signal.y2

            inputs = ImageUtils.preprocess(image[y1:y2, x1:x2], size=64, padding_color=0, precision=self.precision)

            outputs = self.session.run(None, {
                'images': inputs,
            })
            outputs = outputs[0].squeeze()
            outputs = np.argmax(outputs)

            signal.shape = self.get_shape(outputs)

        return signals
