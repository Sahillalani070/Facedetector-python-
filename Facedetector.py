import cv2

face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)

while True:
    frame_read, frame = cam.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coordinates = face_data.detectMultiScale(gray_scale)
    
    for (x,y,w,v) in coordinates:
        cv2.rectangle(frame, (x,y),(x+w, y+v),(0,256,0))
    
    cv2.imshow('Face Detected',frame)
    key =cv2.waitKey(1)
    
    if key ==32:
        break
    
