import cv2
import numpy as np

from utils import ImageUtils


class ShapeClassifier:

    def __init__(self):
        self.model = cv2.dnn.readNetFromONNX('./classifier/weights/classify.onnx')

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

            inputs = ImageUtils.preprocess(image[y1:y2, x1:x2], size=64, padding_color=0)
            self.model.setInput(inputs)

            outputs = self.model.forward()
            outputs = outputs.squeeze()

            shape_index = np.argmax(outputs)
            signal.shape = self.get_shape(shape_index)

        return signals
