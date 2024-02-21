import cv2
import os

from ultralytics import YOLO
from classify.utils import ClassifyUtils
from entity import TrafficLight

class ShapeClassifier:

    def __init__(self, model_path: str = "./classify/weights/best_openvino_model") -> None:
        self.model = YOLO(model_path)

    def __call__(self, image: cv2.Mat, traffic_light_set: set[TrafficLight]) -> set[TrafficLight]:
        return self.classify(image, traffic_light_set)

    def classify(self, image: cv2.Mat, traffic_light_set: set[TrafficLight]) -> set[TrafficLight]:
        for traffic_light in traffic_light_set:
            x1, y1, x2, y2 = traffic_light.rect_xyxy
            result = self.model(ClassifyUtils.letterbox(image[y1:y2, x1:x2]), imgsz = 64, verbose = False)[0]
            traffic_light.shape = result.names[result.probs.top1]
        return traffic_light_set

    def test(self, image_path: str = "./classify/images", result_path: str = "./classify/results") -> None:
        for image_name in os.listdir(image_path):
            image = cv2.imread(f"{image_path}/{image_name}")
            result = self.model(image, imgsz = 64)[0]
            cv2.putText(image, result.names[result.probs.top1], (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
            cv2.imwrite(f"{result_path}/result_{image_name}", image)

if __name__ == "__main__":
    ShapeClassifier().test()
