import cv2
import numpy as np
import os
import time


class PlottingUtils:

    @staticmethod
    def get_color(color_name):
        if color_name == 'red':
            return 0, 0, 255
        if color_name == 'green':
            return 0, 255, 0
        if color_name == 'yellow':
            return 0, 204, 255
        return 127, 127, 127

    @staticmethod
    def plot_traffic_signal(image, signal):
        color = PlottingUtils.get_color(signal.color)

        x1 = signal.x1
        y1 = signal.y1
        x2 = signal.x2
        y2 = signal.y2

        image = cv2.putText(image, signal.shape, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color)

        return image

    @staticmethod
    def plot_traffic_signals(image, signals):
        for signal in signals:
            image = PlottingUtils.plot_traffic_signal(image, signal)
        return image

    @staticmethod
    def plot_direct_rule(image, direct, offset, allow):
        if allow:
            color = PlottingUtils.get_color('green')
        else:
            color = PlottingUtils.get_color('red')

        return cv2.putText(image, direct, (offset, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    @staticmethod
    def plot_passing_rules(image, rules):
        image = PlottingUtils.plot_direct_rule(image, 'Left', 5, rules.left)
        image = PlottingUtils.plot_direct_rule(image, 'Straight', 55, rules.straight)
        image = PlottingUtils.plot_direct_rule(image, 'Right', 145, rules.right)
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

    @staticmethod
    def iter_images(source_path):
        for image_name in os.listdir(source_path):
            yield image_name, cv2.imread(f'{source_path}/{image_name}')

    @staticmethod
    def save_image(image, image_name, result_path):
        cv2.imwrite(f'{result_path}/result_{image_name}', image)


class ResultUtils:

    @staticmethod
    def non_max_suppression(outputs, conf_threshold, nms_threshold):
        classes = list()
        boxes = list()
        scores = list()

        for result in outputs:
            _, max_score, _, max_location = cv2.minMaxLoc(result[4:])

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


class TimingUtils:

    @staticmethod
    def execute_time(function, *args, **kwargs):
        time_tick1 = time.perf_counter()
        result = function(*args, **kwargs)
        time_tick2 = time.perf_counter()
        return result, (time_tick2 - time_tick1)
