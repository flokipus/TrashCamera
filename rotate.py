import cv2
import numpy as np
import imutils


def test_thresholds(img):
    tr_imgs = {}
    n = 10
    delta = 255 // n
    for i in range(n + 1):
        t, tr_img = cv2.threshold(img, i * delta, 255, cv2.THRESH_BINARY)
        tr_imgs[t] = tr_img

    for t, tr_img in tr_imgs.items():
        cnts = cv2.findContours(tr_img, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        colored_img = cv2.cvtColor(tr_img, cv2.COLOR_GRAY2BGR)
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(colored_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow(f"Thresholded[{t}]", colored_img)

    cv2.imshow("InitialGreyImage", img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


WIDTH = 500

if __name__ == "__main__":
    img = cv2.imread("./test.jpg")
    img = imutils.resize(img, width=WIDTH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    t, tr_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(tr_img, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    index, area = max(enumerate(map(cv2.contourArea, contours)), key=lambda x: x[1])
    # print(index, contours[index])
    best_contour = contours[index]
    mask = np.full(img.shape, 255, dtype=np.uint8)
    out = np.full(img.shape, 255, dtype=np.uint8)
    cv2.drawContours(mask, contours, index, 0, -1)
    out[mask==0] = tr_img[mask==0]
    cv2.imshow("1", out)
    cv2.imshow("2", img)

    copy_out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    inverted_out = cv2.bitwise_not(out)
    linesP = cv2.HoughLinesP(inverted_out, 1, np.pi / 180, 50, None, WIDTH*(3/4), 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(copy_out, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("3", copy_out)


    while True:
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
