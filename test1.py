# import packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

start = input("Press Enter to start the subsystem...")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries for the ball and hole
ball_lower = (20, 150, 120)  # Yellow
ball_upper = (40, 235, 255)  # Yellow
#ball_lower = (10, 100, 100) # Orange
#ball_upper = (25, 255, 255) # Orange
hole_lower = (90, 10, 40)      # Black (hole, to be adjusted once the actual hole is constructed) 
hole_upper = (130, 40, 130) # Black

pts = deque(maxlen=args["buffer"])
positions = []  # Store (x, y) positions
stopped_counter = 0  # Counter to check if ball stops

# video input handling
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])
time.sleep(1.0)  # Camera warm-up

ball_detected = False
hole_detected = False
ball_moved = False

while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break

    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Ball detection
    ball_mask = cv2.inRange(hsv, ball_lower, ball_upper)
    #ball_mask = cv2.erode(ball_mask, None, iterations=2)
    #ball_mask = cv2.dilate(ball_mask, None, iterations=2)
    ball_cnts = cv2.findContours(ball_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_cnts = imutils.grab_contours(ball_cnts)

    # Hole detection
    hole_mask = cv2.inRange(hsv, hole_lower, hole_upper)
    #hole_mask = cv2.erode(hole_mask, None, iterations=2)
    #hole_mask = cv2.dilate(hole_mask, None, iterations=2)
    hole_cnts = cv2.findContours(hole_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hole_cnts = imutils.grab_contours(hole_cnts)

    if ball_cnts and not ball_detected:
        print("Ball Detected")
        ball_detected = True
        c = max(ball_cnts, key=cv2.contourArea)
        ((ball_x, ball_y), ball_radius) = cv2.minEnclosingCircle(c)
        ball_center = (int(ball_x), int(ball_y))
    if hole_cnts and not hole_detected:
        print("Hole Detected")
        hole_detected = True
        h = max(hole_cnts, key=cv2.contourArea)
        ((hole_x, hole_y), hole_radius) = cv2.minEnclosingCircle(h)
        hole_center = (int(hole_x), int(hole_y))
        if ball_detected:
            #cv2.line(frame, ball_center, hole_center, (0, 255, 0), 2)
            optimal_trajectory = (ball_center, hole_center)
            positions.append(ball_center)
    
    if len(ball_cnts) > 0 and len(hole_cnts) > 0 and ball_detected and hole_detected:
        cv2.circle(hole_mask, hole_center, int(hole_radius), (255, 0, 0), -1)
        cv2.line(hole_mask, optimal_trajectory[0], optimal_trajectory[1], (0, 255, 0), 2)

    cv2.imshow("Frame", hole_mask)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

if not args.get("video", False):
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()

# Show trajectory after ball stops
trajectory_frame = np.zeros_like(frame)
cv2.line(trajectory_frame, optimal_trajectory[0], optimal_trajectory[1], (0, 255, 0), 2)
for i in range(1, len(positions)):
    if positions[i - 1] is not None and positions[i] is not None:
        cv2.line(trajectory_frame, positions[i - 1], positions[i], (0, 0, 255), 2)
cv2.imshow("Trajectory", trajectory_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
