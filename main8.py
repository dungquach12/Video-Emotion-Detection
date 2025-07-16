import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
print("[DEBUG] Loading model...")
model = tf.keras.models.load_model('vgg16_class_model_2.keras') 
print("[DEBUG] Model loaded successfully.")
emotion_labels = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

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

        # Extract the face region of interest (ROI) from the original RGB frame
        roi_color = frame[y:y+h, x:x+w]  # Assuming 'frame' is in RGB (or BGR if using OpenCV)

        # Resize to match your model's input size
        roi_color = cv2.resize(roi_color, (96, 96))

        # Normalize pixel values to [0, 1]
        roi_color = roi_color.astype('float32') / 255.0

        # Add batch dimension
        roi_color = np.expand_dims(roi_color, axis=0)  # Shape becomes (1, 96, 96, 3)

        # Predict emotion
        preds = model.predict(roi_color)
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
