import cv2

class PlottingUtils:

    @staticmethod
    def get_color(color_name: str) -> tuple[int, int, int]:
        if color_name == "red":
            return 0, 0, 255
        if color_name == "green":
            return 0, 255, 0
        if color_name == "yellow":
            return 0, 204, 255
        return 127, 127, 127
    
    @staticmethod
    def plot_traffic_light(image: cv2.Mat, x1: int, y1: int, x2: int, y2: int, color_name: str, shape: str) -> cv2.Mat:
        color = PlottingUtils.get_color(color_name)
        image = cv2.putText(image, shape, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
        return cv2.rectangle(image, (x1, y1), (x2, y2), color)
    
    @staticmethod
    def plot_signal(image: cv2.Mat, text: str, offset_x: int, allow: bool) -> cv2.Mat:
        if allow:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        return cv2.putText(image, text, (offset_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
