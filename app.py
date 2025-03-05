import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (joint)
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Exercise tracker
class ExerciseCounter:
    def __init__(self, exercise_name):
        self.exercise_name = exercise_name
        self.count = 0
        self.stage = None
        
    def update(self, angle, up_thresh, down_thresh):
        if angle > up_thresh:
            self.stage = "up"
        if angle < down_thresh and self.stage == 'up':
            self.stage = "down"
            self.count += 1

# Exercise recommendation system
def recommend_exercises(exercise_name):
    recommendations = {
        "bicep curl": ["Hammer Curl", "Concentration Curl", "Reverse Curl"],
        "squat": ["Lunges", "Leg Press", "Deadlift"],
        "pushup": ["Chest Press", "Tricep Dips", "Shoulder Press"]
    }
    return recommendations.get(exercise_name, ["Keep Going!"])

# Initialize counters for each exercise
bicep_counter = ExerciseCounter("bicep curl")
squat_counter = ExerciseCounter("squat")
pushup_counter = ExerciseCounter("pushup")

# Start capturing video
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
        
        if landmarks:
            # Get coordinates for key landmarks
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y] 
        
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate joint angles
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            knee_angle = calculate_angle(hip, knee, ankle)
            hip_angle = calculate_angle(shoulder, hip, knee)
            shoulder_angle = calculate_angle(hip, shoulder, elbow)

            
            # Bicep Curl Counter
            bicep_counter.update(elbow_angle, up_thresh=160, down_thresh=30)
            # Squat Counter
            squat_counter.update(knee_angle, up_thresh=170, down_thresh=90)
            # Pushup Counter
            pushup_counter.update(shoulder_angle, up_thresh=160, down_thresh=45)
            
            # Display Count
            cv2.putText(image, f"Bicep Curls: {bicep_counter.count}", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Squats: {squat_counter.count}", 
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Pushups: {squat_counter.count}", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Exercise Recommendations
            # Determine current exercise based on angles
            if elbow_angle < 90:
                current_exercise = "bicep curl"
            elif hip_angle < 90:
                current_exercise = "squat"
            elif shoulder_angle < 45 and elbow_angle > 90:
                current_exercise = "pushup"
            else:
                current_exercise = "unknown"

            # Get recommendations based on the identified exercise
            recommendations = recommend_exercises(current_exercise)

            # Display the recommendations on the screen
            cv2.putText(image, f"Try: {', '.join(recommendations)}", 
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display the video feed
        cv2.imshow('Exercise Counter', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()