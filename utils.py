import cv2
import numpy as np


class ImageUtils:

    @staticmethod
    def letterbox(image, size, padding_color):
        current_size = max(image.shape[0], image.shape[1])

        x1 = (current_size - image.shape[1]) >> 1
        y1 = (current_size - image.shape[0]) >> 1

        x2 = x1 + image.shape[1]
        y2 = y1 + image.shape[0]

        background = np.full((current_size, current_size, 3), padding_color, dtype=np.uint8)
        background[y1:y2, x1:x2] = image

        return cv2.resize(background, (size, size))

    @staticmethod
    def convert(image, precision):
        inputs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
        inputs = inputs / 255.0
        inputs = np.expand_dims(inputs, axis=0)

        if precision == 'fp16':
            return inputs.astype(np.float16)
        else:
            return inputs.astype(np.float32)

    @staticmethod
    def preprocess(image, size, padding_color, precision):
        return ImageUtils.convert(ImageUtils.letterbox(image, size, padding_color), precision)


class ResultUtils:

    @staticmethod
    def get_valid_outputs(outputs, conf_threshold):
        valid_outputs = outputs[np.amax(outputs[:, 4:7], axis=1) > conf_threshold]

        boxes = valid_outputs[:, 0:4]
        confidences = valid_outputs[:, 4:7]

        return boxes.astype(np.int32), confidences

    @staticmethod
    def non_max_suppression(outputs, conf_threshold, iou_threshold):
        boxes, confidences = ResultUtils.get_valid_outputs(outputs, conf_threshold)

        scores = np.amax(confidences, axis=1)
        classes = np.argmax(confidences, axis=1)

        boxes[:, 0] -= boxes[:, 2] >> 1
        boxes[:, 1] -= boxes[:, 3] >> 1

        for index in cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold, eta=0.5):
            yield boxes[index], classes[index]


class MarkingUtils:

    @staticmethod
    def signals(image, signals):
        for signal in signals:
            x1 = signal.x1
            y1 = signal.y1
            x2 = signal.x2
            y2 = signal.y2

            if signal.color_index == 0:
                color = (0, 0, 255)
            elif signal.color_index == 1:
                color = (0, 255, 0)
            elif signal.color_index == 2:
                color = (0, 204, 255)
            else:
                color = (127, 127, 127)

            cv2.putText(image, signal.shape, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
            cv2.rectangle(image, (x1, y1), (x2, y2), color)

    @staticmethod
    def directs(image, directs):
        for is_allow, offset in ((directs.left, 20), (directs.straight, 50), (directs.right, 80)):
            if is_allow:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.circle(image, (offset, 20), 12, color, -1, cv2.LINE_AA)
