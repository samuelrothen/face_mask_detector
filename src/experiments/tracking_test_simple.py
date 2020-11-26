import cv2

cap = cv2.VideoCapture(1)

tracker = cv2.TrackerMOSSE_create()
tracker = cv2.TrackerCSRT_create()


def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)


cap_success, img = cap.read()
img = cv2.flip(img, 1)

bbox = cv2.selectROI('Tracking', img, True)
tracker.init(img, bbox)

while True:
    timer = cv2.getTickCount()
    cap_success, img = cap.read()
    img = cv2.flip(img, 1)

    cap_success, bbox = tracker.update(img)

    if cap_success:
        drawBox(img, bbox)
    else:
        cv2.putText(img, str('Tracking lost'), (75, 75),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - timer))
    cv2.putText(img, str(fps), (75, 50),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Tracking', img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
