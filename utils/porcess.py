import cv2
import numpy as np


def letterbox(image, size, padding_color):
    current_size = max(image.shape[0], image.shape[1])

    x1 = (current_size - image.shape[1]) >> 1
    y1 = (current_size - image.shape[0]) >> 1

    x2 = x1 + image.shape[1]
    y2 = y1 + image.shape[0]

    background = np.full((current_size, current_size, 3), padding_color, dtype=np.uint8)
    background[y1:y2, x1:x2] = image

    return cv2.resize(background, (size, size))


def convert_input(image, precision):
    inputs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
    inputs = inputs / 255.0
    inputs = np.expand_dims(inputs, axis=0)

    if precision == 'fp16':
        return {'images': inputs.astype(np.float16)}
    else:
        return {'images': inputs.astype(np.float32)}


def preprocess(image, size, padding_color, precision):
    return convert_input(letterbox(image, size, padding_color), precision)


def get_valid_outputs(outputs, conf_threshold):
    valid_outputs = outputs[np.amax(outputs[:, 4:7], axis=1) > conf_threshold]

    boxes = valid_outputs[:, 0:4]
    confidences = valid_outputs[:, 4:7]

    return boxes.astype(np.int32), confidences


def non_max_suppression(outputs, conf_threshold, iou_threshold):
    boxes, confidences = get_valid_outputs(outputs, conf_threshold)

    scores = np.amax(confidences, axis=1)
    classes = np.argmax(confidences, axis=1)

    boxes[:, 0] -= boxes[:, 2] >> 1
    boxes[:, 1] -= boxes[:, 3] >> 1

    for index in cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold, eta=0.5):
        yield boxes[index], classes[index]
