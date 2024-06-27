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
        image = PlottingUtils.plot_direct_rule(image, 'LEFT', 5, rules.left)
        image = PlottingUtils.plot_direct_rule(image, 'STRAIGHT', 65, rules.straight)
        image = PlottingUtils.plot_direct_rule(image, 'RIGHT', 165, rules.right)
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
    def convert_inputs(image, precision):
        inputs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
        inputs = inputs / 255.0
        inputs = np.expand_dims(inputs, axis=0)

        if precision == 'fp16':
            return inputs.astype(np.float16)
        else:
            return inputs.astype(np.float32)

    @staticmethod
    def preprocess(image, size, padding_color, precision):
        return ImageUtils.convert_inputs(ImageUtils.letterbox(image, size=size, padding_color=padding_color), precision)

    @staticmethod
    def iter_images(source_path):
        for image_name in os.listdir(source_path):
            yield image_name, cv2.imread(f'{source_path}/{image_name}')

    @staticmethod
    def save_image(image, image_name, result_path):
        cv2.imwrite(f'{result_path}/result_{image_name}', image)


class ResultUtils:

    @staticmethod
    def non_max_suppression(outputs, conf_threshold, iou_threshold):
        classes = list()
        boxes = list()
        scores = list()

        for result in outputs.astype(np.float32):
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

        result_indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold, eta=0.5)

        boxes = np.asarray(boxes, dtype=np.int32)
        classes = np.asarray(classes, dtype=np.int32)

        for index in result_indices:
            yield boxes[index], classes[index]


class TimingUtils:

    @staticmethod
    def execute_time(function, *args, **kwargs):
        time_tick1 = time.perf_counter()
        result = function(*args, **kwargs)
        time_tick2 = time.perf_counter()
        return result, (time_tick2 - time_tick1)
