"""Script for running pet detector on the video from webcam or file."""
import argparse
import sys
from pathlib import Path
from time import time

import cv2
import yaml

sys.path.append('src')

from infra.exceptions import NoDetectionError
from services.pet_detector import PetDetectorYolov8


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run et detector on the video from webcam or file.')
    parser.add_argument('--webcam', type=int, help='Id of the webcam device')
    parser.add_argument('--video', type=str, help='Path to the video file.')
    return parser.parse_args()


def main() -> None:
    args = parse()
    with Path('config.yml').open() as stream:
        config = yaml.safe_load(stream)

    pet_detector = PetDetectorYolov8(
        onnx_model_path=config['services']['pet_detector']['model_path'],
        confidence_thresh=config['services']['pet_detector']['confidence_threshold'],
        iou_thresh=config['services']['pet_detector']['iou_threshold'],
    )

    videocap = cv2.VideoCapture(args.webcam) if args.webcam else cv2.VideoCapture(args.video)

    while True:
        start_time = time()
        ret, frame = videocap.read()

        if not ret:
            break
        try:
            detection_result = pet_detector(frame)
        except NoDetectionError:
            pass
        else:
            pet_detector.draw_detections(frame, detection_result)

        fps = int(1 / (time() - start_time))

        cv2.putText(frame, f'FPS: {fps}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
