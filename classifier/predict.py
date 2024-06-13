import cv2
import numpy as np

from utils import ImageUtils


class MainClassifier:

    def __init__(self):
        self.model = cv2.dnn.readNetFromONNX("./classifier/weights/classify.onnx")

    def __call__(self, image, lights):
        return self.classify(image, lights)

    @staticmethod
    def get_shape(shape_index):
        if shape_index == 0:
            return "full"
        if shape_index == 1:
            return "left"
        if shape_index == 2:
            return "right"
        if shape_index == 3:
            return "straight"
        return None

    def classify(self, image, lights):
        for light in lights:
            x1 = light.x1
            y1 = light.y1
            x2 = light.x2
            y2 = light.y2

            inputs = ImageUtils.preprocess(image[y1:y2, x1:x2], size=64, padding_color=0)
            self.model.setInput(inputs)

            outputs = self.model.forward()
            outputs = outputs.squeeze()

            shape_index = np.argmax(outputs)
            light.shape = self.get_shape(shape_index)

        return lights
