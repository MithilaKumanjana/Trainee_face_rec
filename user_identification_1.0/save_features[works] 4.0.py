import cv2
import dlib
import csv
import tkinter as tk
from tkinter import simpledialog
import numpy as np  # Import NumPy for array operations

def extract_face_features():
    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)

    # Load the face detector and shape predictor models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Create an empty list to store face features
    face_features = []

    # Counter for samples
    sample_count = 0

    while sample_count < 1000:
        # Capture frame-by-frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to capture frame from the webcam.")
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = detector(gray)

        # Iterate over detected faces
        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)

            # Extract face features (e.g., distances between landmarks)
            left_eye = landmarks.part(36)
            right_eye = landmarks.part(45)
            nose_tip = landmarks.part(30)
            mouth_left = landmarks.part(48)
            mouth_right = landmarks.part(54)

            eye_distance = right_eye.x - left_eye.x
            nose_to_mouth_distance = mouth_left.y - nose_tip.y

            # Additional features
            mouth_width = mouth_right.x - mouth_left.x
            eyebrow_height = (landmarks.part(19).y - landmarks.part(24).y) + (landmarks.part(18).y - landmarks.part(25).y)
            jaw_width = landmarks.part(16).x - landmarks.part(0).x

            # Store the extracted features in a list
            face_features.append([eye_distance, nose_to_mouth_distance, mouth_width, eyebrow_height, jaw_width])

            # Draw landmarks on the frame for visualization (optional)
            for point in landmarks.parts():
                cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Increment sample count
        sample_count += 1

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

    # Calculate the averages for each feature column
    face_features_array = np.array(face_features)
    feature_averages = np.mean(face_features_array, axis=0)

    # Prompt the user to enter a name for the CSV file
    root = tk.Tk()
    root.withdraw()
    user_name = simpledialog.askstring("Input", "Enter user name for CSV file:")

    # Save the feature averages to a CSV file
    csv_file_path = f"user_csv\{user_name}.csv"
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Eye Distance', 'Nose to Mouth Distance', 'Mouth Width', 'Eyebrow Height', 'Jaw Width'])
        writer.writerow(feature_averages)

    print(f"Feature averages saved to {csv_file_path}")

if __name__ == "__main__":
    extract_face_features()
