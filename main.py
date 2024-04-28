import cv2
import os
import time

from identifier import TrafficSignalIdentifier

class MainProgram:

    def __init__(self) -> None:
        self.identifier = TrafficSignalIdentifier()

    def images_predict(self, image_path: str = "./images", result_path: str = "./results") -> None:
        total_times = 0
        image_count = 0

        for image_index, image_name in enumerate(os.listdir(image_path), start = 0):
            image = cv2.imread(f"{image_path}/{image_name}")
            start_times = time.perf_counter()
            signal = self.identifier(image, plot_result = True)
            end_times = time.perf_counter()
            delta_times = end_times - start_times
            
            if image_index > 0:
                total_times += delta_times
                image_count += 1

            cv2.imwrite(f"{result_path}/result_{image_name}", image)
            print(f"Image: {image_name:<10}{signal}Times: {delta_times:.3f}s")

        print(f"\nAverage Times: {total_times / image_count:.3f}s\n")


if __name__ == "__main__":
    MainProgram().images_predict()
