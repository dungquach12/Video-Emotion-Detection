import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
print("[DEBUG] Loading model...")
model = tf.keras.models.load_model('best_model.keras') 
print("[DEBUG] Model loaded successfully.")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('emotions_demo.mp4')  # Replace with your video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection and model input
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension
        roi_gray = np.expand_dims(roi_gray, axis=0)   # Add batch dimension

        # Predict emotion
        preds = model.predict(roi_gray)
        emotion = emotion_labels[np.argmax(preds)]

        # Display emotion
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Emotion Detection', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
