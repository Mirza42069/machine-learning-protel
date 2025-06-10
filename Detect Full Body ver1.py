from djitellopy import Tello
import cv2
import mediapipe as mp
import math
import time
from ultralytics import YOLO

# Initialize drone
tello = Tello()
tello.connect()
tello.streamon()

# Load YOLOv8 model for human detection
print("Loading YOLOv8 model...")
yolo_model = YOLO('yolov8n.pt')  # You can use yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt for better accuracy

# Mediapipe modules for body part detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


# Function to get center of key landmarks
def get_center(landmarks, indices, image_shape):
    h, w = image_shape[:2]
    points = []
    for i in indices:
        if landmarks[i].visibility > 0.5:
            x = int(landmarks[i].x * w)
            y = int(landmarks[i].y * h)
            points.append((x, y))
    if not points:
        return None
    avg_x = sum([p[0] for p in points]) // len(points)
    avg_y = sum([p[1] for p in points]) // len(points)
    return (avg_x, avg_y)


# Create MediaPipe models for detailed body part analysis
pose = mp_pose.Pose(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    model_complexity=2,
    enable_segmentation=False,
    smooth_landmarks=True
)

hands = mp_hands.Hands(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    max_num_hands=2
)

# Main loop
try:
    frame_count = 0
    while True:
        # Get frame from Tello
        frame = tello.get_frame_read().frame
        if frame is None:
            print("Failed to get frame")
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1

        # Make a copy for visualization
        output_frame = frame.copy()

        # YOLOv8 Human Detection
        results = yolo_model(frame, verbose=False)

        human_detected = False
        human_boxes = []

        # Process YOLO results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Check if it's a person (class_id = 0 in COCO dataset)
                    if class_id == 0 and confidence > 0.5:
                        human_detected = True

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        human_boxes.append((x1, y1, x2, y2, confidence))

                        # Draw bounding box
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Calculate center of bounding box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Draw center point
                        cv2.circle(output_frame, (center_x, center_y), 8, (0, 255, 0), cv2.FILLED)

                        # Add label
                        label = f"Human: {confidence:.2f}"
                        cv2.putText(output_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Only proceed with detailed body part detection if human is detected
        if human_detected:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with pose detection
            pose_results = pose.process(rgb_frame)

            # Process with hand detection
            hands_results = hands.process(rgb_frame)

            # Draw hands if detected
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        output_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Add hand center
                    h, w = output_frame.shape[:2]
                    wrist_x = int(hand_landmarks.landmark[0].x * w)
                    wrist_y = int(hand_landmarks.landmark[0].y * h)
                    cv2.circle(output_frame, (wrist_x, wrist_y), 6, (0, 255, 255), cv2.FILLED)
                    cv2.putText(output_frame, "Hand", (wrist_x + 10, wrist_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw pose landmarks if detected
            if pose_results.pose_landmarks:
                # Draw pose connections
                mp_drawing.draw_landmarks(
                    output_frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # Analyze body parts
                landmarks = pose_results.pose_landmarks.landmark
                h, w = output_frame.shape[:2]

                # Check for arms
                left_arm_visible = all(landmarks[i].visibility > 0.5 for i in [11, 13])
                right_arm_visible = all(landmarks[i].visibility > 0.5 for i in [12, 14])

                # Check for legs with improved detection
                left_leg_visible = all(landmarks[i].visibility > 0.3 for i in [23, 25])
                right_leg_visible = all(landmarks[i].visibility > 0.3 for i in [24, 26])

                # Additional check for feet
                left_foot_visible = landmarks[27].visibility > 0.3 or landmarks[31].visibility > 0.3
                right_foot_visible = landmarks[28].visibility > 0.3 or landmarks[32].visibility > 0.3

                # Consider leg visible if we see either hip+knee OR ankle/foot
                left_leg_visible = left_leg_visible or left_foot_visible
                right_leg_visible = right_leg_visible or right_foot_visible

                # Check for torso
                torso_visible = all(landmarks[i].visibility > 0.5 for i in [11, 12, 23, 24])

                # Analyze and label body parts
                body_parts_detected = []

                # Arms
                if left_arm_visible:
                    arm_center = get_center(landmarks, [11, 13, 15], output_frame.shape)
                    if arm_center:
                        cv2.circle(output_frame, arm_center, 6, (0, 0, 255), cv2.FILLED)
                        cv2.putText(output_frame, "L-Arm", (arm_center[0] + 10, arm_center[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        body_parts_detected.append("Left Arm")

                if right_arm_visible:
                    arm_center = get_center(landmarks, [12, 14, 16], output_frame.shape)
                    if arm_center:
                        cv2.circle(output_frame, arm_center, 6, (0, 0, 255), cv2.FILLED)
                        cv2.putText(output_frame, "R-Arm", (arm_center[0] + 10, arm_center[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        body_parts_detected.append("Right Arm")

                # Legs
                if left_leg_visible:
                    leg_landmarks = []
                    for i in [23, 25, 27, 29, 31]:  # hip, knee, ankle, heel, foot_index
                        if landmarks[i].visibility > 0.3:
                            leg_landmarks.append(i)

                    if leg_landmarks:
                        leg_center_x = sum(int(landmarks[i].x * w) for i in leg_landmarks) // len(leg_landmarks)
                        leg_center_y = sum(int(landmarks[i].y * h) for i in leg_landmarks) // len(leg_landmarks)
                        leg_center = (leg_center_x, leg_center_y)

                        cv2.circle(output_frame, leg_center, 6, (255, 0, 0), cv2.FILLED)
                        cv2.putText(output_frame, "L-Leg", (leg_center[0] + 10, leg_center[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        body_parts_detected.append("Left Leg")

                if right_leg_visible:
                    leg_landmarks = []
                    for i in [24, 26, 28, 30, 32]:  # hip, knee, ankle, heel, foot_index
                        if landmarks[i].visibility > 0.3:
                            leg_landmarks.append(i)

                    if leg_landmarks:
                        leg_center_x = sum(int(landmarks[i].x * w) for i in leg_landmarks) // len(leg_landmarks)
                        leg_center_y = sum(int(landmarks[i].y * h) for i in leg_landmarks) // len(leg_landmarks)
                        leg_center = (leg_center_x, leg_center_y)

                        cv2.circle(output_frame, leg_center, 6, (255, 0, 0), cv2.FILLED)
                        cv2.putText(output_frame, "R-Leg", (leg_center[0] + 10, leg_center[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        body_parts_detected.append("Right Leg")

                # Torso
                if torso_visible:
                    torso_center = get_center(landmarks, [11, 12, 23, 24], output_frame.shape)
                    if torso_center:
                        cv2.circle(output_frame, torso_center, 8, (0, 255, 0), cv2.FILLED)
                        cv2.putText(output_frame, "Torso", (torso_center[0] + 10, torso_center[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        body_parts_detected.append("Torso")

                # Full body check
                full_body_visible = (torso_visible and
                                     (left_arm_visible or right_arm_visible) and
                                     (left_leg_visible or right_leg_visible))

                if full_body_visible:
                    body_parts_detected.append("Full Body")

                # Calculate distance between wrists if both are visible
                if landmarks[15].visibility > 0.5 and landmarks[16].visibility > 0.5:
                    x1, y1 = int(landmarks[15].x * w), int(landmarks[15].y * h)
                    x2, y2 = int(landmarks[16].x * w), int(landmarks[16].y * h)
                    dist = int(math.hypot(x2 - x1, y2 - y1))
                    cv2.line(output_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(output_frame, f"D: {dist}px", ((x1 + x2) // 2, (y1 + y2) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # Display detected body parts list
                parts_text = f"Parts: {', '.join(body_parts_detected)}"
                cv2.putText(output_frame, parts_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        else:
            # No human detected
            cv2.putText(output_frame, "No Human Detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show battery and frame info
        battery_level = tello.get_battery()
        cv2.putText(output_frame, f"Battery: {battery_level}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output_frame, f"Frame: {frame_count}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Detection status
        humans_count = len(human_boxes)
        if humans_count > 0:
            cv2.putText(output_frame, f"Humans Detected: {humans_count}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Tello YOLOv8 + Body Part Detection", output_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Keyboard interrupt detected")
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    # Cleanup
    print("Cleaning up...")
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()
    print("Done!")
