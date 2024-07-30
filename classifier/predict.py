import numpy as np
import onnxruntime as ort

from utils import ImageUtils


class ShapeClassifier:

    def __init__(self, device, precision):
        if device == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(f'classifier/weights/deploy/classify-{precision}.onnx', providers=providers)
        self.precision = precision

    def __call__(self, image, signals):
        for signal in signals:
            x1 = signal.x1
            y1 = signal.y1
            x2 = signal.x2
            y2 = signal.y2

            inputs = ImageUtils.preprocess(image[y1:y2, x1:x2], size=64, padding_color=0, precision=self.precision)

            outputs = self.session.run(None, {
                'images': inputs,
            })
            signal.shape_index = np.argmax(outputs[0].squeeze())

        return signals
