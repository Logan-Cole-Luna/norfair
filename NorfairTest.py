import cv2
import numpy as np
from norfair import Detection, Tracker, Video, draw_points

# Initialize ORB detector
orb = cv2.ORB_create()

# Norfair
video = Video(input_path="B:\\GitHub\\FeaturePointTracking\\TrackingMedia\\F-16Video.mp4")
tracker = Tracker(distance_function="euclidean", distance_threshold=20, use_gpu=True)

for frame in video:
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray_frame, None)

    # Convert keypoints to Norfair detections
    detections = [Detection(np.array([kp.pt])) for kp in keypoints]

    # Update tracker
    tracked_objects = tracker.update(detections=detections)

    # Draw tracked objects
    frame = draw_points(frame, tracked_objects)

    # Optionally draw keypoints
    frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))
    cv2.imshow("Frame", frame)

    # Write the frame with tracked objects
    video.write(frame)
