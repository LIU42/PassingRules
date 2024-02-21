import numpy
import cv2
import math

class TrafficLight:

    def __init__(self, rect_array: numpy.ndarray, color: str) -> None:
        self.center_x, self.center_y, self.width, self.height = rect_array
        self.color = color
        self.shape = None

    def __str__(self) -> str:
        return f"{self.center_x} {self.center_y} {self.width} {self.height} {self.color} {self.shape}"

    def __eq__(self, other: 'TrafficLight') -> bool:
        return self.rect_xywh == other.rect_xywh
    
    def __hash__(self) -> int:
        return hash(self.center_x) + hash(self.center_y) + hash(self.width) + hash(self.height)
    
    @property
    def rect_xywh(self) -> tuple[int, int, int, int]:
        return (int(self.center_x), int(self.center_y), int(self.width), int(self.height))
    
    @property
    def rect_xyxy(self) -> tuple[int, int, int, int]:
        x1 = self.center_x - self.width / 2
        y1 = self.center_y - self.height / 2
        x2 = self.center_x + self.width / 2
        y2 = self.center_y + self.height / 2
        return (int(x1), int(y1), int(x2), int(y2))
    
    def similarity_to(self, other: 'TrafficLight', weight_x: float = 0.05, weight_y: float = 5, weight_size: float = 2) -> float:
        distance_x = weight_x * (self.center_x - other.center_x)
        distance_y = weight_y * (self.center_y - other.center_y)
        distance_size = weight_size * (abs(self.width - other.width) + abs(self.height - other.height))
        return math.sqrt(distance_x ** 2 + distance_y ** 2) + distance_size
    
    @staticmethod
    def similarity(cluster1: set['TrafficLight'], cluster2: set['TrafficLight']) -> float:
        min_distance = math.inf
        for traffic_light1 in cluster1:
            for traffic_light2 in cluster2:
                min_distance = min(traffic_light1.similarity_to(traffic_light2), min_distance)
        return min_distance
    
    @staticmethod
    def center_distance(cluster: set['TrafficLight'], x: float = 320, y: float = 240) -> float:
        average_center_x = numpy.mean([ traffic_light.center_x for traffic_light in cluster ])
        average_center_y = numpy.mean([ traffic_light.center_y for traffic_light in cluster ])
        return math.sqrt((average_center_x - x) ** 2 + (average_center_y - y) ** 2)
    
    def plot(self, image: cv2.Mat) -> cv2.Mat:
        x1, y1, x2, y2 = self.rect_xyxy
        if self.color == "red":
            mark_color = (0, 0, 255)
        elif self.color == "green":
            mark_color = (0, 255, 0)
        elif self.color == "yellow":
            mark_color = (0, 204, 255)
        else:
            mark_color = (127, 127, 127)
        cv2.putText(image, self.shape, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, mark_color)
        cv2.rectangle(image, (x1, y1), (x2, y2), mark_color)
        return image
    
class TrafficSignal:

    def __init__(self, strategy: str = "conservative") -> None:
        if strategy == "radical":
            self.straight = True
            self.left = True
            self.right = True
        else:
            self.straight = False
            self.left = False
            self.right = True
    
    def __str__(self) -> str:
        return f"Straight: {str(self.straight):<8}Left: {str(self.left):<8}Right: {str(self.right):<8}"
    
    def all_allow(self) -> None:
        self.straight = True
        self.left = True
        self.right = True

    def all_forbid(self) -> None:
        self.straight = False
        self.left = False
        self.right = True

    def plot_signal(self, image: cv2.Mat, text: str, offset_x: int, allow: bool) -> cv2.Mat:
        return cv2.putText(image, text, (offset_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if allow else (0, 0, 255), 2)
    
    def plot(self, image: cv2.Mat) -> cv2.Mat:
        self.plot_signal(image, "Straight", 55, self.straight)
        self.plot_signal(image, "Left", 5, self.left)
        self.plot_signal(image, "Right", 145, self.right)
        return image
    