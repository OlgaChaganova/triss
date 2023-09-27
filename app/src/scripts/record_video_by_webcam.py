"""Script for recording video with webcam."""
import argparse
from pathlib import Path

import cv2


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run et detector on the video from webcam or file.')
    parser.add_argument('webcam', type=int, help='Id of the webcam device')
    parser.add_argument('filename', type=str, help='Name of the file')
    parser.add_argument('--save', type=str, default='../records/', help='Path to the folder where record will be saved')
    parser.add_argument('--show', action='store_true', help='Show video on the display')
    return parser.parse_args()


def record(video_cap: cv2.VideoCapture, video_writer: cv2.VideoWriter, show: bool) -> None:  # noqa: FBT001
    while True:
        ret, frame = video_cap.read()

        if not ret:
            break
        video_writer.write(frame)

        if show:
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video_cap.release()
    video_writer.release()

    if show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse()

    filepath = str(Path(args.save) / args.filename)

    if not Path(args.save).exists():
        Path(args.save).mkdir(parents=True)

    video_cap = cv2.VideoCapture(args.webcam)
    video_writer = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640, 480))

    try:
        record(video_cap=video_cap, video_writer=video_writer, show=args.show)
    except KeyboardInterrupt:
        video_cap.release()
        video_writer.release()

    print(f'The video was successfully saved: {filepath}')  # noqa: T201
