import cv2
from random import randrange as r

frontal_face_file = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_file = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

webcam = cv2.VideoCapture("happy.mp4")

#Loop forever
while True:
    frame_reads,frame =webcam.read() #reading frames through web cam
    #Converting frames into graycolour frame
    gray_face_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Detecting face from capture video
    detect_face = frontal_face_file.detectMultiScale(gray_face_frame)
    #Draw reactangle over the face
    for (x,y,w,h) in detect_face:
        cv2.rectangle(frame,(x,y),(x+w,y+h+30),(r(255),r(255),r(255)),4)
        # Now detecting the smile on selected area of a frame
        the_face = frame[y:y+h,x:x+w]
        smile_gray = gray_face_frame = cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)
        smile_detect = smile_file.detectMultiScale(smile_gray,scaleFactor=1.7, minNeighbors=20)
        #Looping through the smile cordinates
        for (x1,y1,w1,h1) in detect_face:
            #Labling the smile on face
            #cv2.rectangle(the_face,(x1,y1),(x1+w1,y1+h1),(0,255,0),4)
            if len(smile_detect)>0:
                cv2.putText(frame,"SMILING",(x,y+h+80),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
        
    cv2.imshow("Smile_Detection",frame)
    k = cv2.waitKey(1) 
    if k == 82:
        break
    
#clean up     
webcam.release()
cv2.destroyAllWindows()      
    
   
    
    
    
    
    
    
#print("done")