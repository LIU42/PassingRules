import cv2
import os
import statistics
import time

from identifier import TrafficSignalIdentifier

def predict_images(identifier: TrafficSignalIdentifier, images_path: str = "./images", result_path: str = "./results") -> None:
    cost_times = list()
    for image_name in os.listdir(images_path):
        image = cv2.imread(f"{images_path}/{image_name}")

        entry_time = time.perf_counter()
        signal = identifier(image)
        leave_time = time.perf_counter()

        cost_time = leave_time - entry_time
        cost_times.append(cost_time)

        cv2.imwrite(f"{result_path}/result_{image_name}", image)
        print(f"Image: {image_name:<10} {signal} Times: {cost_time:.3f}s")

    print(f"Average Times: {statistics.mean(sorted(cost_times)[1:-1]):.3f}s")


if __name__ == "__main__":
    identifier = TrafficSignalIdentifier()
    predict_images(identifier)
