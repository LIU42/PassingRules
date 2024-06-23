import argparse
import cv2
import os
import statistics

from recognition import RulesRecognizer
from utils import TimingUtils


def recognize_images(recognizer, source_path, result_path):
    execute_times = list()

    for image_name in os.listdir(source_path):
        image = cv2.imread(f'{source_path}/{image_name}')
        rules, execute_time = TimingUtils.execute_time(recognizer, image)

        cv2.imwrite(f'{result_path}/result_{image_name}', image)
        execute_times.append(execute_time)

        print(f'Image: {image_name:<10} {rules} Times: {execute_time:.3f}s')

    print(f'Average Times: {statistics.mean(sorted(execute_times)[1:-1]):.3f}s')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_path', type=str, default='./images')
    parser.add_argument('--result_path', type=str, default='./results')

    parser.add_argument('--conf_threshold', type=float, default=0.25)
    parser.add_argument('--nms_threshold', type=float, default=0.45)

    parser.add_argument('--filter_weights', type=tuple, default=(0.05, 5, 2))
    parser.add_argument('--filter_threshold', type=float, default=40)

    parser.add_argument('--strategy', type=str, default='conservative', choices=['conservative', 'radical'])
    parser.add_argument('--plotting', type=bool, default=True)

    arguments = parser.parse_args()
    recognizer = RulesRecognizer(
        conf_threshold=arguments.conf_threshold,
        nms_threshold=arguments.nms_threshold,
        filter_threshold=arguments.filter_threshold,
        filter_weights=arguments.filter_weights,
        strategy=arguments.strategy,
        plotting=arguments.plotting,
    )
    recognize_images(recognizer, arguments.source_path, arguments.result_path)


if __name__ == '__main__':
    main()
