import numpy as np
import onnxruntime as ort

import utils.porcess as process


class ShapeClassifier:
    def __init__(self, configs):
        if configs['device'] == 'CUDA':
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.configs = configs
        self.session = ort.InferenceSession(f'classify/weights/deploy/classify-{self.precision}.onnx', providers=providers)

    def __call__(self, image, signals):
        for signal in signals:
            x1 = signal.x1
            y1 = signal.y1
            x2 = signal.x2
            y2 = signal.y2

            inputs = process.preprocess(image[y1:y2, x1:x2], size=64, padding_color=0, precision=self.precision)

            outputs = self.session.run(None, inputs)
            outputs = self.reshape(outputs)

            signal.shape_index = np.argmax(outputs)

        return signals
    
    @property
    def precision(self):
        return self.configs['precision']

    @staticmethod
    def reshape(outputs):
        return outputs[0].squeeze()
