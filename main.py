import time
import datetime

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream


MIN_AREA = 5

WINDOW_NAME = "ABC"


if __name__ == "__main__":
    vs = VideoStream(src=0).start()
    firstFrame = None
    firstFrameTime = time.time()
    kernel = np.ones((3, 3), 'uint8')
    while True:
        # grab the current frame and initialize the occupied/unoccupied
        # text
        frame = vs.read()
        text = "Unoccupied"
        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if frame is None:
            print("Error: we got if frame is None:")
            break
        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        grayed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayed_frame = cv2.GaussianBlur(grayed_frame, (21, 21), 0)
        # if the first frame is None, initialize it
        timeNow = time.time()
        if firstFrame is None or (timeNow - firstFrameTime) > 1.0:
            firstFrame = grayed_frame
            firstFrameTime = timeNow
            continue

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, grayed_frame)
        thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("Initial Thresh", thresh)
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, kernel, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < MIN_AREA:
                continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"

        # draw the text and timestamp on the frame
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
        # cleanup the camera and close any open windows
    vs.stop()
    cv2.destroyAllWindows()
