import cv2
import os

from ultralytics import YOLO
from classify.utils import ClassifyUtils
from entities import TrafficLight

class ShapeClassifier:

    def __init__(self, model_path: str = "./classify/weights/best_openvino_model") -> None:
        self.model = YOLO(model_path)

    def __call__(self, image: cv2.Mat, traffic_light_set: set[TrafficLight]) -> set[TrafficLight]:
        return self.classify(image, traffic_light_set)
    
    def get_image_range(self, image: cv2.Mat, x1: int, y1: int, x2: int, y2: int) -> cv2.Mat:
        return ClassifyUtils.letterbox(image[y1:y2, x1:x2])

    def classify(self, image: cv2.Mat, traffic_light_set: set[TrafficLight]) -> set[TrafficLight]:
        for traffic_light in traffic_light_set:
            result = self.model(self.get_image_range(image, *traffic_light.rect_xyxy), imgsz = 64, verbose = False)[0]
            traffic_light.shape = result.names[result.probs.top1]
        return traffic_light_set
    
    def mark_result(self, image: cv2.Mat, label: str) -> cv2.Mat:
        cv2.putText(image, label, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))

    def test(self, image_path: str = "./classify/images", result_path: str = "./classify/results") -> None:
        for image_name in os.listdir(image_path):
            image = cv2.imread(f"{image_path}/{image_name}")
            result = self.model(image, imgsz = 64)[0]
            cv2.imwrite(f"{result_path}/result_{image_name}", self.mark_result(image, result.names[result.probs.top1]))


if __name__ == "__main__":
    ShapeClassifier().test()
