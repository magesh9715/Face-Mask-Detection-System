# Importing Libraries
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Loading the Model
model = load_model('mask_detector.h5')

# Loading the face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initiating the video capture from file
video_path = "sgg.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit()

# Process video frame by frame
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or error reading frame")
        break
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, 1.2, 7, minSize=(60, 60),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) == 0:
        print("No faces detected in this frame")
        continue
    
    # Initialize count lists
    correct_mask_count = []
    no_mask_count = []
    incorrect_mask_count = []
    
    # Process each detected face
    for (x, y, w, h) in faces:
        cropped_face = frame[y:y + h, x:x + w]
        face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)
        
        pred = model.predict(face, verbose=0)
        (WithoutMask, CorrectMask, InCorrectMask) = pred[0]
        
        # Determine mask status
        if max(pred[0]) == CorrectMask:
            label = "Correct Mask"
            color = (0, 255, 0)
            correct_mask_count.append(1)
        elif max(pred[0]) == InCorrectMask:
            label = "Incorrect Mask"
            color = (0, 165, 255)
            incorrect_mask_count.append(2)
        else:
            label = "No Mask"
            color = (0, 0, 255)
            no_mask_count.append(0)
        
        # Add confidence percentage
        label = "{}: {:.2f}%".format(label, max(WithoutMask, CorrectMask, InCorrectMask) * 100)
        
        # Draw rectangle and text on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display statistics
    face_count = len(no_mask_count) + len(incorrect_mask_count) + len(correct_mask_count)
    stats_text = f"Faces: {face_count} | No Mask: {len(no_mask_count)} | Incorrect: {len(incorrect_mask_count)} | Correct: {len(correct_mask_count)}"
    cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show the output frame
    cv2.imshow("Face Mask Detection", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
