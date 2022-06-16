import cv2
from cv2 import dilate
import numpy as np

#video from camera
cap= cv2.VideoCapture('video.mp4')

#retangulo
min_width_react = 80  
min_height_react = 80

#line counter
count_line_position = 550

#initialize substructor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    x1= int(w/2)
    y1= int(h/2)
    cx= x+x1
    cy= y+y1
    return cx,cy

detect = [] 
offset= 6 #allowable error between pixel
count = 0

while True:
    ret,frame1 = cap.read()
    grey = cv2.cvtColor (frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)

    #applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterSahpe,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,0),3)

    for (i,c) in enumerate(counterSahpe):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>= min_width_react) & (h>= min_height_react)
        if not validate_counter:
            continue

        cv2.rectangle((frame1),(x,y),(x+h,y+h),(0,255,0),2)

        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255), -1)	

        for (x,y) in detect:
            if y < (count_line_position + offset) and y > (count_line_position - offset):
                count += 1
            
            cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,127,255),3)
            detect.remove((x,y))

            #print("Vehicle counter: " + str(count))
         

    cv2.putText(frame1,"CONTADOR DE VEICULOS :"+ str(count), (80,70) , cv2.FONT_HERSHEY_COMPLEX ,2, (0,20,255), 5)

    #cv2.imshow('Dectecter', dilatada)
 
    cv2.imshow('Video Original', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
cap.release()
