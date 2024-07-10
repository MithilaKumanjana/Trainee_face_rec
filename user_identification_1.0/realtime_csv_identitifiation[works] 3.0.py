import cv2
import dlib
import csv
import os

def load_all_saved_features(folder_path):
    # Load face features from all CSV files in the specified folder
    all_saved_features = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            csv_file = os.path.join(folder_path, filename)
            with open(csv_file, 'r') as file:
                reader = csv.reader(file)
                header = next(reader)  # Skip the header row
                for row in reader:
                    all_saved_features.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), os.path.splitext(filename)[0]])
    return all_saved_features

# def compare_face_features(saved_features, current_features, threshold=90.0):
#     # Compare the current face features with the saved features
#     for idx, saved_feature in enumerate(saved_features):
#         eye_distance_diff = abs(saved_feature[0] - current_features[0])
#         nose_to_mouth_diff = abs(saved_feature[1] - current_features[1])
#         mouth_width_diff = abs(saved_feature[2] - current_features[2])
#         eyebrow_height_diff = abs(saved_feature[3] - current_features[3])
#         jaw_width_diff = abs(saved_feature[4] - current_features[4])

#         if eye_distance_diff < threshold and nose_to_mouth_diff < threshold and \
#             mouth_width_diff < threshold and eyebrow_height_diff < threshold and jaw_width_diff < threshold:
#             return saved_feature[5]  # Return the name of the matched user
#     return None
# def compare_face_features(saved_features, current_features, threshold=5.0):
#     # Compare the current face features with the saved features
#     for idx, saved_feature in enumerate(saved_features):
#         eye_distance_diff = abs(saved_feature[0] - current_features[0])
#         nose_to_mouth_diff = abs(saved_feature[1] - current_features[1])
#         mouth_width_diff = abs(saved_feature[2] - current_features[2])
#         eyebrow_height_diff = abs(saved_feature[3] - current_features[3])
#         jaw_width_diff = abs(saved_feature[4] - current_features[4])

#         print(f"User {idx + 1}: Eye Diff: {eye_distance_diff}, Nose-Mouth Diff: {nose_to_mouth_diff}, Mouth Width Diff: {mouth_width_diff}, Eyebrow Height Diff: {eyebrow_height_diff}, Jaw Width Diff: {jaw_width_diff}")

#         if eye_distance_diff < threshold and nose_to_mouth_diff < threshold and \
#             mouth_width_diff < threshold and eyebrow_height_diff < threshold and jaw_width_diff < threshold:
#             return saved_feature[5]  # Return the name of the matched user
#     return None

def compare_face_features(saved_features, current_features, threshold=30.0):
    # Compare the current face features with the saved features
    for idx, saved_feature in enumerate(saved_features):
        eye_distance_diff = abs(saved_feature[0] - current_features[0])
        nose_to_mouth_diff = abs(saved_feature[1] - current_features[1])
        mouth_width_diff = abs(saved_feature[2] - current_features[2])
        eyebrow_height_diff = abs(saved_feature[3] - current_features[3])
        jaw_width_diff = abs(saved_feature[4] - current_features[4])

        print(f"User {idx + 1}: Eye Diff: {eye_distance_diff}, Nose-Mouth Diff: {nose_to_mouth_diff}, Mouth Width Diff: {mouth_width_diff}, Eyebrow Height Diff: {eyebrow_height_diff}, Jaw Width Diff: {jaw_width_diff}")

        if eye_distance_diff < threshold and nose_to_mouth_diff < threshold and \
            mouth_width_diff < threshold and eyebrow_height_diff < threshold and jaw_width_diff < threshold:
            return saved_feature[5]  # Return the name of the matched user

    # Return None if all 5 features do not match
    return None



def recognize_faces(saved_features, webcam_id=0):
    # Initialize the webcam
    video_capture = cv2.VideoCapture(webcam_id)

    # Load the face detector and shape predictor models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to capture frame from the webcam.")
            break
        # Flip the frame horizontally
        #frame = cv2.flip(frame, 1)

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
            mouth_width = mouth_right.x - mouth_left.x
            eyebrow_height = (landmarks.part(19).y - landmarks.part(24).y) + (landmarks.part(18).y - landmarks.part(25).y)
            jaw_width = landmarks.part(16).x - landmarks.part(0).x

            # Compare face features with saved features
            current_features = [eye_distance, nose_to_mouth_distance, mouth_width, eyebrow_height, jaw_width]
            matched_user = compare_face_features(saved_features, current_features)

            # Draw a bounding box around the face and display the name if a match is found
            if matched_user is not None:
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, matched_user, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    csv_folder_path = r"user_csv"  # Update with the path to your folder containing CSV files
    saved_features = load_all_saved_features(csv_folder_path)
    recognize_faces(saved_features)
