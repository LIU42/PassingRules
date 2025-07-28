import argparse
import cv2
import inferences.engines as engines


def put_bbox(image, detection_result):
    detection_bbox, detection_label = detection_result

    x1 = detection_bbox[0]
    y1 = detection_bbox[1]
    x2 = detection_bbox[2]
    y2 = detection_bbox[3]

    if detection_label[1] == '0':
        return cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 215), thickness=2)
    else:
        return cv2.rectangle(image, (x1, y1), (x2, y2), (0, 215, 0), thickness=2)


def put_text(image, detection_result):
    detection_bbox, detection_label = detection_result

    x1 = detection_bbox[0]
    y1 = detection_bbox[1]

    if detection_label[1] == '0':
        return cv2.putText(image, detection_label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 215), thickness=2)
    else:
        return cv2.putText(image, detection_label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 0), thickness=2)


def put_rule(image, position, result):
    if result:
        return cv2.circle(image, position, 12, (0, 215, 0), thickness=-1)
    else:
        return cv2.circle(image, position, 12, (0, 0, 215), thickness=-1)


def main(args):
    sources = args.sources
    outputs = args.outputs

    for source, output in zip(sources, outputs):
        image = cv2.imread(source)
        results, detections = engines.inference(image)

        for detection in detections:
            image = put_bbox(image, detection)
            image = put_text(image, detection)

        image = put_rule(image, (20, 20), results[0])
        image = put_rule(image, (50, 20), results[1])
        image = put_rule(image, (80, 20), results[2])

        cv2.imwrite(output, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--sources', nargs='+', required=True)
    parser.add_argument('-o', '--outputs', nargs='+', required=True)

    main(parser.parse_args())
