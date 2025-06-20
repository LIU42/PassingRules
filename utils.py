import cv2


def draw_bound(image, detection_result):
    detection_bbox, detection_label = detection_result

    x1 = detection_bbox[0]
    y1 = detection_bbox[1]
    x2 = detection_bbox[2]
    y2 = detection_bbox[3]

    if detection_label[1] == '0':
        return cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 215), thickness=2)
    else:
        return cv2.rectangle(image, (x1, y1), (x2, y2), (0, 215, 0), thickness=2)


def draw_label(image, detection_result):
    detection_bbox, detection_label = detection_result

    x1 = detection_bbox[0]
    y1 = detection_bbox[1]

    if detection_label[1] == '0':
        return cv2.putText(image, detection_label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 215), thickness=2)
    else:
        return cv2.putText(image, detection_label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 0), thickness=2)


def draw_result(image, position, result):
    if result:
        return cv2.circle(image, position, 12, (0, 215, 0), thickness=-1)
    else:
        return cv2.circle(image, position, 12, (0, 0, 215), thickness=-1)
