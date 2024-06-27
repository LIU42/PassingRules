import argparse
import statistics

from recognition import RulesRecognizer
from utils import ImageUtils
from utils import TimingUtils


def recognize_images(recognizer, source_path, result_path):
    execute_times = list()

    for image_name, image in ImageUtils.iter_images(source_path):
        rules, execute_time = TimingUtils.execute_time(recognizer, image)
        execute_times.append(execute_time)
        ImageUtils.save_image(image, image_name, result_path)

        print(f'Image: {image_name:<10} {rules} Times: {execute_time:.3f}s')

    print(f'Average Times: {statistics.mean(sorted(execute_times)[1:-1]):.3f}s')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_path', type=str, default='./images')
    parser.add_argument('--result_path', type=str, default='./results')

    parser.add_argument('--device', type=str, default='CPU', choices=['CPU', 'GPU'])
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16'])

    parser.add_argument('--conf_threshold', type=float, default=0.25)
    parser.add_argument('--iou_threshold', type=float, default=0.45)

    parser.add_argument('--filter_weights', type=tuple, default=(0.05, 5, 2))
    parser.add_argument('--filter_threshold', type=float, default=40)

    parser.add_argument('--strategy', type=str, default='conservative', choices=['conservative', 'radical'])
    parser.add_argument('--plotting', type=bool, default=True)

    args = parser.parse_args()
    recognizer = RulesRecognizer(
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
        precision=args.precision,
        filter_threshold=args.filter_threshold,
        filter_weights=args.filter_weights,
        strategy=args.strategy,
        plotting=args.plotting,
    )
    recognize_images(recognizer, args.source_path, args.result_path)


if __name__ == '__main__':
    main()
