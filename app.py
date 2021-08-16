import imutils
import numpy as np
import cv2
from model.Net import Net
from tensorflow import keras

net = Net()
print("Completed training phase.")

faceCascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
cam.set(3,640) # set Width
cam.set(4,480) # set Height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

id = 0

while True:
    ret, img =cam.read()
    img = imutils.resize(img, width=800)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) ## buat kotak di sekitar wajah yang terdeteksi
        resized_face = cv2.resize(gray[y:y+h,x:x+w], (100,100))
        np_face = keras.preprocessing.image.img_to_array(resized_face)
        np_face = np.expand_dims(np_face, axis=0)
        prediction = net.predict(np_face)
        ## tulis hasil prediksi dan confidence di atas kotak
        cv2.putText(
            img, 
            net.target_classes[prediction], 
            (x,y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255,255,255), 
            2
        )

    cv2.imshow('Face Recognition',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()