import cv2
import numpy as np
from tensorflow.keras.models import load_model
model = load_model(r"C:\Users\me\Desktop\dataset\asl\face_emotion_1.h5")
# Load the cascade
face_cascade = cv2.CascadeClassifier(r"C:\Users\me\Desktop\dataset\asl\haarcascade_frontalface_default.xml")

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')
classes = {'0': 'angry',
 '1': 'disgust',
 '2': 'fear',
 '3': 'happy',
 '4': 'neutral',
 '5': 'sad',
 '6': 'surprise'}
 
def predict(img):
    img = cv2.resize(img, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_p = img.reshape(1, 48, 48, 3).astype('float32')/255.0
    pred = model.predict(img_p, verbose=0)
    label = np.argmax(pred)
    return classes[str(label)]

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    # Draw the rectangle around each face
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            label = predict(img[y:y+h, x:x+w])
            cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    # Display
    if img.shape[0] > 10 and img.shape[1] > 10:
        cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object