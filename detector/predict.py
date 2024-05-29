import cv2
from ultralytics import YOLO

from entities import TrafficLight

class TrafficLightDetector:

    def __init__(self, model_path: str = "./detector/weights/detect_openvino_model") -> None:
        self.model = YOLO(model_path, task="detect")

    def __call__(self, image: cv2.Mat) -> list[TrafficLight]:
        return self.detect(image)

    def detect(self, image: cv2.Mat) -> list[TrafficLight]:
        result = self.model(image, verbose=False)[0].numpy()
        detected_list = list()
        for box_index, classes_index in enumerate(result.boxes.cls, start=0):
            detected_list.append(TrafficLight(*result.boxes.xywh[box_index], result.names[classes_index]))
        return detected_list
