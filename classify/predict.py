import numpy
import cv2
import os

from ultralytics import YOLO
from entity import TrafficLight

class ShapeClassifier:

    def __init__(self, model_path: str = "./classify/weights/best.pt") -> None:
        self.model = YOLO(model_path)

    def __call__(self, image: cv2.Mat, traffic_light_set: set[TrafficLight]) -> set[TrafficLight]:
        return self.classify(image, traffic_light_set)

    def letterbox(self, image: cv2.Mat, new_size: int = 64) -> cv2.Mat:
        aspect_ratio = image.shape[1] / image.shape[0]
        if image.shape[1] > image.shape[0]:
            image_resize = cv2.resize(image, (new_size, int(new_size / aspect_ratio)))
        else:
            image_resize = cv2.resize(image, (int(new_size * aspect_ratio), new_size))

        background = numpy.zeros((new_size, new_size, 3), dtype = numpy.uint8)
        x = (new_size - image_resize.shape[1]) // 2
        y = (new_size - image_resize.shape[0]) // 2
        background[y:y + image_resize.shape[0], x:x + image_resize.shape[1]] = image_resize
        return background

    def classify(self, image: cv2.Mat, traffic_light_set: set[TrafficLight]) -> set[TrafficLight]:
        for traffic_light in traffic_light_set:
            x1, y1, x2, y2 = traffic_light.rect_xyxy
            result = self.model(self.letterbox(image[y1:y2, x1:x2]), verbose = False)[0]
            traffic_light.shape = result.names[result.probs.top1]
        return traffic_light_set

    def test(self, image_path: str = "./classify/images", result_path: str = "./classify/results") -> None:
        for image_name in os.listdir(image_path):
            image = cv2.imread(f"{image_path}/{image_name}")
            result = self.model(image)[0]
            cv2.putText(image, result.names[result.probs.top1], (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
            cv2.imwrite(f"{result_path}/result_{image_name}", image)

if __name__ == "__main__":
    ShapeClassifier().test()
