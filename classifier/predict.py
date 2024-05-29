import cv2
from ultralytics import YOLO

from entities import TrafficLight
from utils import ImageUtils

class ShapeClassifier:

    def __init__(self, model_path: str = "./classifier/weights/classify_openvino_model") -> None:
        self.model = YOLO(model_path, task="classify")

    def __call__(self, image: cv2.Mat, traffic_lights: set[TrafficLight]) -> set[TrafficLight]:
        return self.classify(image, traffic_lights)

    def classify(self, image: cv2.Mat, traffic_lights: set[TrafficLight]) -> set[TrafficLight]:
        for traffic_light in traffic_lights:
            result = self.model(ImageUtils.cut(image, *traffic_light.box_xyxy), imgsz=64, verbose=False)[0]
            traffic_light.shape = result.names[result.probs.top1]
        return traffic_lights
