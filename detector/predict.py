import cv2

from structures import TrafficLight
from utils import ImageUtils
from utils import ResultUtils


class TrafficLightDetector:

    def __init__(self):
        self.model = cv2.dnn.readNetFromONNX("./detector/weights/detect.onnx")

    def __call__(self, image):
        return self.detect(image)

    @staticmethod
    def get_color(color_index):
        if color_index == 0:
            return "red"
        if color_index == 1:
            return "green"
        if color_index == 2:
            return "yellow"
        return None

    def detect(self, image):
        inputs = ImageUtils.preprocess(image, size=640, padding_color=127)
        self.model.setInput(inputs)

        outputs = self.model.forward()
        outputs = outputs.squeeze()
        outputs = cv2.transpose(outputs)

        boxes, classes = ResultUtils.non_max_suppression(outputs, conf_threshold=0.25, nms_threshold=0.45)
        detections = list()

        for box, color_index in zip(boxes, classes):
            detections.append(TrafficLight(box[0], box[1] - 80, box[2], box[3], self.get_color(color_index)))

        return detections
