import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model_path = 'D:/164 Project/model_weights.keras'  # Ensure to update this path to where your model is stored
model = load_model(model_path)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the list of emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            fc = gray[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))  # Resize face to expected input size
            roi = np.expand_dims(roi, axis=-1)  # Add channel dimension
            roi = np.expand_dims(roi, axis=0)  # Add batch dimension

            pred = model.predict(roi)
            emotion_idx = np.argmax(pred)
            emotion_text = emotions[emotion_idx]

            # Display the emotion text and a bounding box around the face
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
