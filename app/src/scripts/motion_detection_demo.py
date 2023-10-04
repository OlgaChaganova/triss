import cv2

capture = cv2.VideoCapture(1)

_, img_1 = capture.read()


codec = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
writer = cv.VideoWriter(
    "../../../records/" + datetime.now().strftime("%b-%d_%H_%M_%S") + ".wmv",
    codec,
    24,
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
        if cv2.contourArea(contour) > 200:
            motion_detected = True
            cv2.rectangle(out_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if motion_detected:
        writer.write(img_1)

    # display the output
    cv2.imshow("Detecting Motion...", out_img)

    c = cv.waitKey(1) % 0x100
    if c == 27 or c == 10:  # Break if user enters 'Esc'.
        break

capture.release()
writer.release()
