
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time



print("Resolution?(144/480/720/1080(not recommended))")
resolution = int(input())



face_frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
face_profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')



capture = cv2.VideoCapture(0)
if(resolution == 1080):
    #1080p
    capture.set(3, 1920)
    capture.set(4, 1080)

if(resolution == 480):
    #480p
    capture.set(3,854) 
    capture.set(4,480)

if(resolution == 144):
    #144p
    capture.set(3, 192)
    capture.set(4, 144)

if(resolution == 720):
    #720p
    capture.set(3, 1280)
    capture.set(4, 720)



new_frame_time = 0
prev_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX



while True:
    ret, raw = capture.read()
    raw = cv2.flip(raw, 1)
    monochrome = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    hue = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    #mask = cv2.inRange(hue, Orange_LB,Orange_UB)
    gray = raw
    
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    fps = int(fps)
    fps = str(fps)
    cv2.putText(raw, fps, (10, 25), font, 0.75, (0, 0, 0), 1, cv2.LINE_AA)


    
    #Face Frontal, Torso Extrapolated
    face_frontal_detect = face_frontal_cascade.detectMultiScale(
        monochrome,
        scaleFactor=1.3,
        minNeighbors=5,      
        minSize=(30, 30)
    )
    
    for (x,y,w,h) in face_frontal_detect:
        cv2.rectangle(raw,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = monochrome[y:y+h, x:x+w]
        roi_color = raw[y:y+h, x:x+w]
        roi_hsv = hue[y:y+h, x:x+w]
        width=w//4
        height = h//4
        cv2.rectangle(raw, (x-2*width, y+6*height), (x+6*width, y+4*h),(255,0,0),2)
        #cv2.circle(raw, (x+2*width, y+3*h),15,(0,0,255),1)

    
    
    #Upper Body/Torso detection --> WIP
    '''upperbody_detect = upperbody_cascade.detectMultiScale(
        monochrome,
        scaleFactor= 1.3,
        minNeighbors= 5,
        minSize=(30,30)
    )

    for (tx, ty, tw, th) in upperbody_detect:
        cv2.rectangle(raw,(tx,ty),(tx+tw, ty+th),(0,0,255),2)'''
        
        
    
    #Smile --> WIP
    '''smile_detect = smile_cascade.detectMultiScale(
            monochrome,
            scaleFactor= 1.5,
            minNeighbors=5,
            minSize=(5, 5),
            )
        
    for (ex, ey, ew, eh) in smile_detect:
            cv2.rectangle(raw, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            print(ex, ey, ew, eh)
               
        
    smile_detect2 = smile_cascade.detectMultiScale(
            monochrome,
            scaleFactor= 1.5,
            minNeighbors=15,
            minSize=(25, 25),
            )
        
    for (xx, yy, ww, hh) in smile_detect2:
            cv2.rectangle(raw, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)'''
        
    
    
    
    cv2.imshow('video', raw)
    #cv2.imshow('HSV', hue)
    #cv2.imshow("HSV Orange masked",mask)

    k = cv2.waitKey(30) & 0xff
    if k == 27: #press 'ESC' to quit
        break

capture.release()
cv2.destroyAllWindows()