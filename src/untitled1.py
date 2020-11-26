import cv2
import numpy as np


img = cv2.imread('test.png')


imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)


img_edges = cv2.Canny(img, 200, 255)
kernel=np.ones((5,5))
img_dil=cv2.dilate(img_edges,kernel,iterations=1)


# getContours(img_dil,imgContour)

contours,hier=cv2.findContours(img_dil,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


cv2.imshow('test',img_dil)
