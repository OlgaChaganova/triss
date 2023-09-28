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
    parser.add_argument('--fps',type=int, default=30, help='FPS')
    parser.add_argument('--img_width', type=int, default=1280, help='Frame width')
    parser.add_argument('--img_height', type=int, default=720, help='Frame height')

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

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    video_cap = cv2.VideoCapture()
    video_cap.open(args.webcam)
    video_cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.img_width)
    video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.img_height)
    video_cap.set(cv2.CAP_PROP_FPS, args.fps)

    video_writer = cv2.VideoWriter(
        filepath, fourcc, args.fps, (args.img_width, args.img_height),
    )#  (1280, 720))

    try:
        record(video_cap=video_cap, video_writer=video_writer, show=args.show)
    except KeyboardInterrupt:
        video_cap.release()
        video_writer.release()

    print(f'The video was successfully saved: {filepath}')  # noqa: T201
