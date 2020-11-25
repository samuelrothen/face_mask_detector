import cv2
import numpy as np

framewidth=640
frameheight=480

cap = cv2.VideoCapture(0)
cap.set(3,framewidth)
cap.set(4,frameheight)





success,img=cap.read()

# imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# array_images=[[img,img,img],
#               [imgGray,img,img]]


def stackCV2Images(array_images):
    hstack_img=[]
    for i_row,images_row in enumerate(array_images):
        for i_col,image in enumerate(images_row):
            if len(image.shape)==2:
                array_images[i_row][i_col]=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        hstack_img.append(np.hstack(images_row))
    vstack_img=np.vstack(hstack_img)
    return vstack_img

img_edges=cv2.Canny(img,100,100)

cv2.imshow('Result',stackCV2Images([[img,img_edges]]))


#%%

def nothing(x):
    pass

cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Trackbars',640,240)
cv2.createTrackbar('TH1','Trackbars',150,500,nothing)
cv2.createTrackbar('TH2','Trackbars',255,500,nothing)


while True:
    success,img=cap.read()
    img = cv2.flip(img, 1)
    
    imgBlur=cv2.GaussianBlur(img,(7,7),1)
    imgGray=cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY)
    
    th1=cv2.getTrackbarPos('TH1','Trackbars')
    th2=cv2.getTrackbarPos('TH2','Trackbars')
    img_edges=cv2.Canny(img,th1,th2)

    
    
    cv2.imshow('Result',stackCV2Images([[img,imgGray,img_edges]]))
    if cv2.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()