import cv2
from datetime import datetime
import argparse
from pathlib import Path
import os


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run video recorder with motion detection.')
    parser.add_argument('webcam', type=int, default=0, help='Id of the webcam device')
    parser.add_argument('--out', type=str, default='../records/', help='Path to the folder where record will be saved')
    parser.add_argument('--show', action='store_true', help='Show video on the display')
    parser.add_argument('--fps', type=int, default=30, help='FPS')
    parser.add_argument('--threshold', type=int, default=100, help='Motion detection threshold')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    capture = cv2.VideoCapture(args.webcam)

    filepath = str(Path(args.save))

    if not Path(args.save).exists():
        Path(args.save).mkdir(parents=True)

    _, img_1 = capture.read()
    img_2 = None

    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    writer = cv2.VideoWriter(
        os.path.join(filepath, datetime.now().strftime("%b-%d_%H_%M_%S") + ".wmv"),
        codec,
        args.fps,
        img_1.shape[1::-1],
        1)

    while capture.isOpened():
        img_2 = img_1
        _, img_1 = capture.read()

        diff = cv2.absdiff(img_1, img_2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh_bin = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        out_img = img_1.copy()
        motion_detected = False
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.contourArea(contour) > args.threshold:
                motion_detected = True
                cv2.rectangle(out_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if motion_detected:
            writer.write(img_1)

        if args.show:
            cv2.imshow("Detecting Motion...", out_img)

        c = cv2.waitKey(1) % 0x100
        if c == 27 or c == 10:  # Break if user enters 'Esc'.
            break

    capture.release()
    writer.release()
