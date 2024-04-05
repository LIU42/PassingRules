import cv2
import os

from ultralytics import YOLO
from entities import TrafficLightBuilder
from entities import TrafficLight

class TrafficLightDetector:

    def __init__(self, model_path: str = "./detect/weights/best_openvino_model") -> None:
        self.model = YOLO(model_path)

    def __call__(self, image: cv2.Mat) -> list[TrafficLight]:
        return self.detect(image)

    def detect(self, image: cv2.Mat) -> list[TrafficLight]:
        result = self.model(image, verbose = False)[0]
        detected_list = list[TrafficLight]()
        for index, classes_index in enumerate(result.boxes.cls.tolist(), start = 0):
            detected_list.append(TrafficLightBuilder.from_xywh_array(result.boxes.xywh[index].numpy(), result.names[classes_index]))
        return detected_list
    
    def test(self, image_path: str = "./detect/images", result_path: str = "./detect/results") -> None:
        for image_name in os.listdir(image_path):
            image = cv2.imread(f"{image_path}/{image_name}")
            result = self.model(image)[0]
            cv2.imwrite(f"{result_path}/result_{image_name}", result.plot())

if __name__ == "__main__":
    TrafficLightDetector().test()
