import cv2
import numpy

class ClassifyUtils:

    @staticmethod
    def letterbox(image: cv2.Mat, new_size: int = 64) -> cv2.Mat:
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
