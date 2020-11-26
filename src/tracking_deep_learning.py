import cv2
import numpy as np

framewidth = 640
frameheight = 480

cap = cv2.VideoCapture(0)
cap.set(3, framewidth)
cap.set(4, frameheight)

success, img = cap.read()


def sliding_window(image, step, ws):
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            print(x, y)
            # return (x, y, image[y:y + ws[1], x:x + ws[0]])


image = sliding_window(img, 8, [100, 100])


# %%
cv2.imshow('Result', img)

cv2.imshow('Window', image[2])
