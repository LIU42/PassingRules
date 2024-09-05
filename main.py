import cv2
import os
import statistics
import time
import yaml
import utils.paint as paint

from recognize import RulesRecognizer


def load_recognizer():
    with open('configs/config.yaml', 'r') as configs:
        return RulesRecognizer(yaml.safe_load(configs))


def load_images():
    for image_name in os.listdir('examples/sources'):
        yield image_name, cv2.imread(f'examples/sources/{image_name}')


def save_result(image, image_name, results):
    signals, directs = results

    paint.signals(image, signals)
    paint.directs(image, directs)

    cv2.imwrite(f'examples/outputs/result_{image_name}', image)


def execute_recognition(recognizer, image):
    counter1 = time.perf_counter()
    results = recognizer(image)
    counter2 = time.perf_counter()

    return results, counter2 - counter1


def average_time(execution_times):
    return statistics.mean(sorted(execution_times)[1:-1])


def main():
    recognizer = load_recognizer()
    execution_times = []

    for image_name, image in load_images():
        results, execution_time = execute_recognition(recognizer, image)

        execution_times.append(execution_time)
        save_result(image, image_name, results)

        print(f'image: {image_name:<8} time: {execution_time:.3f}s')

    print(f'average time: {average_time(execution_times):.3f}s')


if __name__ == '__main__':
    main()
