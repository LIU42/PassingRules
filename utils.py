import cv2
import numpy as np

from structures import TrafficLight
from structures import TrafficSignal


class PlottingUtils:

    @staticmethod
    def get_color(color_name):
        if color_name == "red":
            return 0, 0, 255
        if color_name == "green":
            return 0, 255, 0
        if color_name == "yellow":
            return 0, 204, 255
        return 127, 127, 127

    @staticmethod
    def plot_traffic_light(image, light):
        color = PlottingUtils.get_color(light.color)

        x1 = light.x1
        y1 = light.y1
        x2 = light.x2
        y2 = light.y2

        image = cv2.putText(image, light.shape, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color)

        return image

    @staticmethod
    def plot_traffic_lights(image, lights):
        for light in lights:
            image = PlottingUtils.plot_traffic_light(image, light)
        return image

    @staticmethod
    def plot_signal(image, text, offset, allow):
        if allow:
            color = PlottingUtils.get_color("green")
        else:
            color = PlottingUtils.get_color("red")

        return cv2.putText(image, text, (offset, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    @staticmethod
    def plot_traffic_signal(image, signal):
        image = PlottingUtils.plot_signal(image, "Left", 5, signal.left)
        image = PlottingUtils.plot_signal(image, "Straight", 55, signal.straight)
        image = PlottingUtils.plot_signal(image, "Right", 145, signal.right)
        return image


class ImageUtils:

    @staticmethod
    def letterbox(image, size, padding_color):
        current_size = max(image.shape[0], image.shape[1])
        x1 = (current_size - image.shape[1]) // 2
        y1 = (current_size - image.shape[0]) // 2

        x2 = x1 + image.shape[1]
        y2 = y1 + image.shape[0]

        background = np.full((current_size, current_size, 3), padding_color, dtype=np.uint8)
        background[y1:y2, x1:x2] = image

        return cv2.resize(background, (size, size))

    @staticmethod
    def blob(image, size):
        return cv2.dnn.blobFromImage(image, scalefactor=(1 / 255), size=(size, size), swapRB=True)

    @staticmethod
    def preprocess(image, size, padding_color):
        return ImageUtils.blob(ImageUtils.letterbox(image, size=size, padding_color=padding_color), size=size)


class ResultUtils:

    @staticmethod
    def non_max_suppression(outputs, conf_threshold, nms_threshold):
        classes = list()
        boxes = list()
        scores = list()

        for result in outputs:
            min_score, max_score, min_location, max_location = cv2.minMaxLoc(result[4:])

            if max_score > conf_threshold:
                boxes.append([
                    result[0] - result[2] * 0.5,
                    result[1] - result[3] * 0.5,
                    result[2],
                    result[3],
                ])
                scores.append(max_score)
                classes.append(max_location[1])

        result_index = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold, eta=0.5)
        result_boxes = np.array(boxes, dtype=np.int32)[result_index]
        result_classes = np.array(classes, dtype=np.int32)[result_index]

        return result_boxes, result_classes
