import cv2
import os
import statistics
import time

from identify import TrafficSignalIdentifier


def predict_images(identifier, images_path, result_path):
    exec_times = list()

    for image_name in os.listdir(images_path):
        image = cv2.imread(f"{images_path}/{image_name}")

        entry_time = time.perf_counter()
        signal = identifier(image)
        leave_time = time.perf_counter()

        exec_time = leave_time - entry_time
        exec_times.append(exec_time)

        cv2.imwrite(f"{result_path}/result_{image_name}", image)
        print(f"Image: {image_name:<10} {signal} Times: {exec_time:.3f}s")

    print(f"Average Times: {statistics.mean(sorted(exec_times)[1:-1]):.3f}s")


if __name__ == "__main__":
    predict_images(TrafficSignalIdentifier(), images_path="./images", result_path="./results")
