# import packages
from collections import deque
import numpy as np
import cv2
import imutils
import time

start = input("Press Enter to start the subsystem...")

# define the lower and upper boundaries for the ball and hole
ball_lower = (20, 150, 120)  # Yellow
ball_upper = (40, 235, 255)  # Yellow
#ball_lower = (0, 180, 150) # Orange
#ball_upper = (20, 225, 255) # Orange
hole_lower = (88, 50, 45)      # Black (hole, to be adjusted once the actual hole is constructed) 
hole_upper = (120, 90, 150) # Black
#hole_lower = (90, 10, 40)      # Black (hole, to be adjusted once the actual hole is constructed) 
#hole_upper = (130, 40, 130) # Black

green_lower = (60, 150, 20)
green_upper = (95, 255, 150)

pts = deque(maxlen=1)
positions = []  # Store (x, y) positions
stopped_counter = 0  # Counter to check if ball stops

# video input handling
vs = cv2.VideoCapture(0)
time.sleep(1.0)
vs.set(cv2.CAP_PROP_FPS, 60)
'''
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
'''

vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
'''

'''

ball_detected = False
hole_detected = False
ball_moved = False
rectangle_detected = False

print(f"Requested FPS: 60, Got {vs.get(cv2.CAP_PROP_FPS)}")

while True:
    ret, frame = vs.read()

    if not rectangle_detected:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        #Green Rectangle
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_corners = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Ignore small noise
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

                if len(approx) == 4:  # We want exactly 4 corners (rectangle)
                    if area > max_area:  # Pick the largest valid rectangle
                        max_area = area
                        best_corners = np.array([point[0] for point in approx], dtype=np.float32)

        if best_corners is not None:
            min_values = np.min(best_corners.astype(int), axis=0)
            max_values = np.max(best_corners.astype(int), axis=0)
            rectangle_detected = True
            #print(max_values - min_values)
    
    if rectangle_detected:
        frame = frame.copy()[min_values[1]:max_values[1], min_values[0]:max_values[0]]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Ball detection
        ball_mask = cv2.inRange(hsv, ball_lower, ball_upper)
        #ball_mask = cv2.erode(ball_mask, None, iterations=2)
        #ball_mask = cv2.dilate(ball_mask, None, iterations=2)
        ball_cnt = cv2.findContours(ball_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ball_cnts = imutils.grab_contours(ball_cnt)

        # Hole detection
        hole_mask = cv2.inRange(hsv, hole_lower, hole_upper)
        hole_mask = cv2.GaussianBlur(hole_mask, (9, 9), 2)
        hole_mask = cv2.erode(hole_mask, None, iterations=2)
        hole_mask = cv2.dilate(hole_mask, None, iterations=2)

        if ball_cnts and not ball_detected:
            print("Ball Detected")
            ball_detected = True
            c = max(ball_cnts, key=cv2.contourArea)
            ((ball_x, ball_y), ball_radius) = cv2.minEnclosingCircle(c)
            ball_center = (int(ball_x), int(ball_y))
        
        # Detect circles using Hough Circle Transform
        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                param1=50, param2=30, minRadius=10, maxRadius=100)

        # Find the largest detected circle
        largest_circle = None
        max_radius = 0

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")  # Convert to integer values
            
            for (x, y, r) in circles:
                #cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.circle(mask, (x, y), r, 255, -1)
                
                mean_val = cv2.mean(hole_mask, mask)
                #print(mean_val)
                
                if mean_val[0] > 180 and r > max_radius:  # Ensure it's dark and the largest circle
                    max_radius = r
                    largest_circle = (x, y, r)

            # Draw the largest detected hole
            #if largest_circle is not None:
            if largest_circle is not None and not hole_detected:
                hole_x, hole_y, hole_radius = largest_circle
                hole_center = (hole_x, hole_y)
                print("Hole Detected")
                hole_detected = True
                #cv2.circle(frame, hole_center, hole_radius, (0, 255, 0), 2)  # Draw detected hole
                if ball_detected:
                    cv2.line(frame, ball_center, hole_center, (0, 255, 0), 2)
                    optimal_trajectory = (ball_center, hole_center)
                    positions.append(ball_center)

        center = None
        if len(ball_cnts) > 0 and ball_detected and hole_detected: 
            optimal_trajectory = (ball_center, hole_center)
            cv2.line(frame, optimal_trajectory[0], optimal_trajectory[1], (0, 255, 0), 2)
            cv2.circle(frame, hole_center, int(hole_radius), (255, 0, 0), 2)

            c = max(ball_cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if center is not None and positions[-1] is not None:
                if radius > 0:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                # Ball starts moving
                if not ball_moved and len(positions) > 0 and np.linalg.norm(np.array(center) - np.array(positions[-1])) > 10:  
                    print("Ball is hit!")
                    ball_moved = True
                '''
                # Ball stops moving
                if len(positions) > 0 and np.linalg.norm(np.array(center) - np.array(positions[-1])) < 1 and ball_moved:  
                    stopped_counter = stopped_counter + 1
                
                if stopped_counter > 120:  # Ball stopped moving
                    print("Ball Stopped. Subsystem Stopping...")
                    break'''
                if len(positions) > 30 and ball_moved:
                    distances = np.linalg.norm(np.array(positions[-30:]) - np.array(center), axis=1)
                    print(distances)
                    if np.mean(distances) < 1:
                        print("Ball Stopped. Subsystem Stopping...")
                        if np.linalg.norm(np.array(center) - np.array(hole_center)) < 1:
                            print("Target hit")
                        else:
                            print("Target missed")
                        break

            positions.append(center)  # Record position
            #print(" ", ball_center, " ", center)

        # Draw the actual trajectory
        if len(positions) > 1:
            for i in range(1, len(positions)):
                if positions[i - 1] is not None and positions[i] is not None:
                    cv2.line(frame, positions[i - 1], positions[i], (0, 0, 255), 2)

        pts.appendleft(center)
        #for i in range(1, len(pts)):
        if pts[0] is not None:
            cv2.line(frame, pts[0], hole_center, (0, 255, 255), 2)

        cv2.imshow("Frame", frame)
        cv2.setWindowProperty("Frame", cv2.WND_PROP_TOPMOST, 1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

vs.release()
#vs.stop()
cv2.destroyAllWindows()

# Show trajectory after ball stops
trajectory_frame = np.zeros_like(frame)
cv2.line(trajectory_frame, optimal_trajectory[0], optimal_trajectory[1], (0, 255, 0), 2)
for i in range(1, len(positions)):
    if positions[i - 1] is not None and positions[i] is not None:
        cv2.line(trajectory_frame, positions[i - 1], positions[i], (0, 0, 255), 2)
cv2.imshow("Trajectory", trajectory_frame)
cv2.setWindowProperty("Trajectory", cv2.WND_PROP_TOPMOST, 1)
cv2.waitKey(0)
cv2.destroyAllWindows()
