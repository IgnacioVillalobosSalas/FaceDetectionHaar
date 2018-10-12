# -*- coding: utf-8 -*-
"""
@author: Ignacio Villalobos Salas
"""

import cv2


#Loading Haar cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')



def detect(grey_frame, frame):
    faces = face_cascade.detectMultiScale(grey_frame,1.3,5)
    for (x,y,w,h) in faces:
        # Frame, top-left corner, bottom-right, rgb and thickness
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_grey = grey_frame[y:y+h,x:x+w]
        roi_frame = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey,1.4,10)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        smiles = smile_cascade.detectMultiScale(roi_grey,1.7,30)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_frame,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    return frame


#Webcam
video = cv2.VideoCapture(0)
while True:
    #last frame
    _, frame = video.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(grey,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

