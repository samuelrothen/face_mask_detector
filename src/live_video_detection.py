import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import mobilenet_v2



def returnDetectedFaces(img,model_face,th=0.9):
    #Hight and Width of the Image (in Pixel)
    img_h, img_w = img.shape[:2]
    #Blob Preprocessing
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),(104, 177, 123))
    #Use the Blob to create Face Predictions
    model_face.setInput(blob)
    detections = model_face.forward()
    #Prepare empty Lists for the Images of the Faces
    #Coordinates of the Bounding-Boxes (+ Center)
    #Detection-Probabilities
    l_faces = []
    l_coords = []
    l_probas = []
    l_centers=[]
    #Looping over all the detected Faces (ordered by Probability)
    for i in range(detections.shape[2]):
        proba = detections[0, 0, i, 2]
        #Stop the Loop if Probability reaches the Threshold
        if proba<th:
            break
        else:
            #Extracting the coordinates (%) and calculating the Position in Pixels
            coordinates=detections[0, 0, i, 3:7]
            x1,y1,x2,y2=(coordinates*[img_w,img_h,img_w,img_h]).astype("int")
            #Check if the Positions are inside the given Image-Dimensions
            x1=max(0, x1)
            y1=max(0, y1)
            x2=min(img_w-1,x2)
            y2=min(img_h-1,y2)
            #Calculate the Center of the Face
            x_center=int((x1+x2)/2)
            y_center=int((y1+y2)/2)
            #Extracting the Image of the detected Face
            face = img[y1:y2, x1:x2]
            #Appending the Results to the Lists
            l_faces.append(face)
            l_coords.append((x1,y1,x2,y2))
            l_probas.append(proba)
            l_centers.append((x_center,y_center))
    return l_faces,l_coords,l_probas,l_centers


def preprocessImage(img):
    #Preprocessing Steps before the Mask-Prediction
    prep_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    prep_img = cv2.resize(prep_img, (224, 224))
    prep_img=mobilenet_v2.preprocess_input(prep_img)
    prep_img=np.expand_dims(prep_img, axis=0)
    return prep_img


def returnProbaMask(face,model_mask):
    #Returns the Probability for Mask / no Mask from Face-Input-Img
    pred_mask = model_mask.predict(preprocessImage(face))
    prob_mask=pred_mask[0,0]
    prob_no_mask=pred_mask[0,1]
    return prob_mask,prob_no_mask


#Pretrained cv2-Model for Face-Recognition
model_face = cv2.dnn.readNet('../models/deploy.prototxt',
                          '../models/res10_300x300_ssd_iter_140000.caffemodel')


#Load the Mask Prediction Model
model_mask = keras.models.load_model('../models/mask_detection_model2.h5')


#Initialize Capture and set Framesize
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)


while True:
    #Get current Image and flip it
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    #Return the detected Faces
    l_faces,l_coords,l_probas,l_centers=returnDetectedFaces(img,model_face,th=0.6)

    #Loop over detected Faces to check for Mask
    for i,face in enumerate(l_faces):
        coords=l_coords[i]
        #Return the Probas for Mask / no Mask
        prob_mask,prob_no_mask=returnProbaMask(face,model_mask)
        #Set Colors and Textlabels according to Probas
        if prob_mask>prob_no_mask:
            color=(0, 255, 0)
            text=f'Mask: {round(prob_mask*100,1)}%'
        else:
            color=(0, 0, 255)
            text=f'No Mask: {round(prob_no_mask*100,1)}%'
        
        #Draw Rectangle around Faces and Label them
        cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]),
                      color, 2)
        cv2.putText(img, text, (coords[0], coords[1] - 10),
                     cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 1)
        cv2.putText(img, f'FaceDetect: {round(l_probas[i]*100,1)}%', (coords[0], coords[1] - 40),
                     cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 1)
        # cv2.circle(img, l_centers[i], 10,
        #                color, 2)
        
    cv2.imshow('VideoCapture', img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


