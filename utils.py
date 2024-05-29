import cv2
import numpy

class PlottingUtils:

    @staticmethod
    def get_color(color_name: str) -> tuple[int, int, int]:
        if color_name == "red":
            return (0, 0, 255)
        if color_name == "green":
            return (0, 255, 0)
        if color_name == "yellow":
            return (0, 204, 255)
        return (127, 127, 127)
    
    @staticmethod
    def plot_traffic_light(image: cv2.Mat, x1: int, y1: int, x2: int, y2: int, color_name: str, shape: str) -> cv2.Mat:
        color = PlottingUtils.get_color(color_name)
        image = cv2.putText(image, shape, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
        return cv2.rectangle(image, (x1, y1), (x2, y2), color)
    
    @staticmethod
    def plot_signal(image: cv2.Mat, text: str, offset_x: int, allow: bool) -> cv2.Mat:
        return cv2.putText(image, text, (offset_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if allow else (0, 0, 255), 2)


class ImageUtils:

    @staticmethod
    def letterbox(image: cv2.Mat, new_size: int = 64) -> cv2.Mat:
        aspect_ratio = image.shape[1] / image.shape[0]
        if image.shape[1] > image.shape[0]:
            image_resize = cv2.resize(image, (new_size, int(new_size / aspect_ratio)))
        else:
            image_resize = cv2.resize(image, (int(new_size * aspect_ratio), new_size))

        background = numpy.zeros((new_size, new_size, 3), dtype=numpy.uint8)
        x = (new_size - image_resize.shape[1]) // 2
        y = (new_size - image_resize.shape[0]) // 2
        background[y:y + image_resize.shape[0], x:x + image_resize.shape[1]] = image_resize
        
        return background
    
    @staticmethod
    def cut(image: cv2.Mat, x1: int, y1: int, x2: int, y2: int) -> cv2.Mat:
        return ImageUtils.letterbox(image[y1:y2, x1:x2])
