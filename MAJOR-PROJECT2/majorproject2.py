#IMAGE PROCESSING PROJECT
import cv2
cap = cv2.VideoCapture('project.mp4') 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #FACE DETECTION MODEL
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') #EYE DETECTION MODEL
while True:
    ret,img = cap.read() #READING THE VIDEO
    if type(img) == type(None):
        break
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #GRAYSCALE IMAGE

    faces = face_cascade.detectMultiScale(gray,1.1,4) #SCALEFACTOR AND NEIGHBORS
    eyes = eye_cascade.detectMultiScale(gray,1.1,18)  #SCALEFACTOR AND NEIGHBORS#SCALEFACTOR AND NEIGHBORS

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
    for (x,y,w,h) in eyes:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
    cv2.imshow('Video',img)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
