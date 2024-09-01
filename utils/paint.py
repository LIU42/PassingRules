import cv2


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


def directs(image, directs):
    for is_allow, offset in ((directs.left, 20), (directs.straight, 50), (directs.right, 80)):
        if is_allow:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.circle(image, (offset, 20), 12, color, -1, cv2.LINE_AA)
