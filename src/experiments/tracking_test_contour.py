import cv2
import numpy as np

framewidth = 640
frameheight = 480

cap = cv2.VideoCapture(0)
cap.set(3, framewidth)
cap.set(4, frameheight)

success, img = cap.read()


def stackCV2Images(array_images):
    hstack_img = []
    for i_row, images_row in enumerate(array_images):
        for i_col, image in enumerate(images_row):
            if len(image.shape) == 2:
                array_images[i_row][i_col] = cv2.cvtColor(
                    image, cv2.COLOR_GRAY2BGR)
        hstack_img.append(np.hstack(images_row))
    vstack_img = np.vstack(hstack_img)
    return vstack_img


img_edges = cv2.Canny(img, 100, 100)

cv2.imshow('Result', stackCV2Images([[img, img_edges]]))


# %%

def nothing(x):
    pass


cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Trackbars', 640, 240)
cv2.createTrackbar('TH1', 'Trackbars', 150, 500, nothing)
cv2.createTrackbar('TH2', 'Trackbars', 255, 500, nothing)


def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (0, 0, 255), 3, 1)


def getContours(img, imgContour):
    contours, hier = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            cv2.drawContours(imgContour, contour, -1, (255, 0, 0), 7)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            print(len(approx))
            bbox = cv2.boundingRect(approx)

            if bbox[2] < 200:
                drawBox(imgContour, bbox)
                cv2.putText(
                    imgContour,
                    f'Points: {len(approx)}',
                    (bbox[0],
                     bbox[1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0,
                     0,
                     255),
                    2)
                # cv2.putText(imgContour, f'Area: {area}',(bbox[0], bbox[1]-30),
                #     cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(
                    imgContour,
                    f'W/H: {bbox[2]}/{bbox[3]}',
                    (bbox[0],
                     bbox[1] - 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0,
                     0,
                     255),
                    2)


success, img1 = cap.read()
success, img2 = cap.read()


lower_color = (166, 50, 50)
upper_color = (186, 255, 255)


while True:
    # img=cv2.absdiff(img1,img2)
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgContour = img.copy()

    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    th1 = cv2.getTrackbarPos('TH1', 'Trackbars')
    th2 = cv2.getTrackbarPos('TH2', 'Trackbars')
    img_edges = cv2.Canny(img, th1, th2)
    kernel = np.ones((5, 5))
    img_dil = cv2.dilate(img_edges, kernel, iterations=1)

    getContours(img_dil, imgContour)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)

    cv2.imshow('Result', stackCV2Images([[img, imgBlur, imgGray],
                                         [img_edges, img_dil, imgContour]]))
    # cv2.namedWindow('Result', flags=cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow('Result', stackCV2Images([[img_edges,imgContour],[hsv,mask]]))
    # img1=img2
    # success, img2 = cap.read()

    if cv2.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
