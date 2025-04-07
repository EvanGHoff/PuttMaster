# import packages
from collections import deque
import numpy as np
import cv2
import imutils
import pickle
import time
import calibrate
import Add_score

# define the lower and upper boundaries for the ball and hole
ball_lower = (20, 150, 120)  # Yellow
ball_upper = (40, 235, 255)  # Yellow
#ball_lower = (0, 180, 150) # Orange
#ball_upper = (20, 225, 255) # Orange
hole_lower = (85, 73, 35)      # Black (hole, to be adjusted once the actual hole is constructed) 
hole_upper = (120, 160, 160) # Black
#hole_lower = (90, 10, 40)      # Black (hole, to be adjusted once the actual hole is constructed) 
#hole_upper = (130, 40, 130) # Black

green_lower = (60, 150, 20) #paper
green_upper = (95, 255, 150) #paper

pts = deque(maxlen=1)
positions = []  # Store (x, y) positions
#stopped_counter = 0  # Counter to check if ball stops
ball_detected = False
hole_detected = False
ball_moved = False
rectangle_detected = False
ball_out_of_green = False

dst_points = pickle.load(open('Raspberry PI Code/matrixes/dstPTS.p','rb'))
score = 0

vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vs.set(cv2.CAP_PROP_FPS, 60)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

calibrating = False
if calibrating:
    calibrate.main(vs)
print(f"Requested FPS: 60, Got {vs.get(cv2.CAP_PROP_FPS)}")

#time.sleep(1.0)

while True:
    ret, frame = vs.read()

    if not rectangle_detected:  
        if dst_points is None:
            break
        else:
            min_values = np.min(dst_points.astype(int), axis=0)
            max_values = np.max(dst_points.astype(int), axis=0)
            #print("Min:", min_values)
            #print("Max:", max_values)
            rectangle_detected = True
    
    if rectangle_detected:
        frame = frame.copy()[min_values[1]:max_values[1], min_values[0]:max_values[0]]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        corrected_frame = np.zeros((max_values[1] - min_values[1], max_values[0] - min_values[0], 3), dtype=np.uint8)

        #calibrate.rectangle()

        # Ball detection
        ball_mask = cv2.inRange(hsv, ball_lower, ball_upper)
        #ball_mask = cv2.erode(ball_mask, None, iterations=2)
        #ball_mask = cv2.dilate(ball_mask, None, iterations=2)
        ball_cnt = cv2.findContours(ball_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ball_cnts = imutils.grab_contours(ball_cnt)

        # Hole detection
        hole_mask = cv2.inRange(hsv, hole_lower, hole_upper)
        #hole_mask = cv2.GaussianBlur(hole_mask, (9, 9), 2)
        #hole_mask = cv2.erode(hole_mask, None, iterations=2)
        #hole_mask = cv2.dilate(hole_mask, None, iterations=2)

        #cv2.imshow("hole_mask", hole_mask)

        if not ball_detected and ball_cnts:
            print("Ball Detected")
            ball_detected = True
            c = max(ball_cnts, key=cv2.contourArea)
            ((ball_x, ball_y), ball_radius) = cv2.minEnclosingCircle(c)
            ball_center = (int(ball_x), int(ball_y))
            positions.append(ball_center)
        
        # Detect circles using Hough Circle Transform
        if not hole_detected:
            gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                    param1=50, param2=30, minRadius=10, maxRadius=100)

            # Find the largest detected circle
            largest_circle = None
            max_radius = 0

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")  # Convert to integer values
                
                for (x, y, r) in circles:
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                    mask = np.zeros(frame.shape[:2], dtype="uint8")
                    cv2.circle(mask, (x, y), r, 255, -1)
                    
                    mean_val = cv2.mean(gray, mask)
                
                    #print(mean_val, "   ", (x, y, r))
                    if mean_val[0] < 100 and r > max_radius:  # Ensure it's dark and the largest circle
                        max_radius = r
                        largest_circle = (x, y, r)

                # Draw the largest detected hole
                if largest_circle is not None:
                #if largest_circle is not None and not hole_detected:
                    hole_x, hole_y, hole_radius = largest_circle
                    hole_center = (hole_x, hole_y)
                    print("Hole Detected")
                    hole_detected = True
                    cv2.circle(frame, hole_center, int(hole_radius), (255, 0, 0), 2)
                    cv2.circle(corrected_frame, hole_center, int(hole_radius), (255, 0, 0), 2)

        center = None
        if ball_detected and hole_detected: 
            optimal_trajectory = (ball_center, hole_center)
            cv2.line(frame, optimal_trajectory[0], optimal_trajectory[1], (0, 255, 0), 2)
            cv2.line(corrected_frame, optimal_trajectory[0], optimal_trajectory[1], (0, 255, 0), 2)
            cv2.circle(frame, hole_center, int(hole_radius), (255, 0, 0), 2)
            cv2.circle(corrected_frame, hole_center, int(hole_radius), (255, 0, 0), 2)

            if len(ball_cnts) > 0:
                c = max(ball_cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            if center is not None and positions[-1] is not None:
                positions.append(center)
                if radius > 0:
                    cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
                    cv2.circle(corrected_frame, center, int(radius), (0, 255, 255), 2)
                    #cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                # Ball starts moving
                #print(np.linalg.norm(np.array(center) - np.array(positions[-2])) / np.linalg.norm(max_values - min_values))
                #print(np.linalg.norm(np.array(center) - np.array(positions[-2])))
                if not ball_moved and np.linalg.norm(np.array(center) - np.array(positions[-2])) / np.linalg.norm(max_values - min_values) > 0.03:
                    print("Ball is hit!")
                    ball_moved = True

            elif center is None and positions[-1] is not None:
                positions.append(positions[-1]) 

        # Draw the actual trajectory
        if len(positions) > 1:
            for i in range(1, len(positions)):
                if positions[i - 1] is not None and positions[i] is not None:
                    cv2.line(frame, positions[i - 1], positions[i], (0, 0, 255), 2)
                    cv2.line(corrected_frame, positions[i - 1], positions[i], (255, 255, 255), 2)

        #case 1, stops on the green
        if len(positions) > 240 and ball_moved:
            distances = np.linalg.norm(np.array(positions[-240:]) - np.array(center), axis=1)
            print(np.mean(distances))
            if 0 < np.mean(distances) / np.linalg.norm(max_values - min_values) < 0.005:
                print("Ball Stopped. Subsystem Stopping...")
                '''
                if np.linalg.norm(np.array(center) - np.array(hole_center)) < 1:
                    print("Target hit")
                else:
                    print("Target missed")'''
                break
            #case 2, out of green
            edge_threshold = 20
            near_edge_count = sum(
            Add_score.is_near_edge(pos, dst_points, edge_threshold) 
            for pos in positions[-240]
            )
            if near_edge_count > 235 and 0 < np.mean(distances) / np.linalg.norm(max_values - min_values) == 0:
                ball_out_of_green = True

        pts.appendleft(center)
        if pts[0] is not None:
            cv2.line(frame, pts[0], hole_center, (0, 255, 255), 2)
            cv2.line(corrected_frame, pts[0], hole_center, (0, 255, 255), 2)

            if ball_out_of_green:
                Add_score.add_score_to_image(corrected_frame, score)
                break
            else:
                score = Add_score.get_Score(np.array(pts[0]), np.array(hole_center), dst_points, score)
                Add_score.add_score_to_image(corrected_frame, score)
        else:
            Add_score.add_score_to_image(corrected_frame, score)
        
        resized_img = cv2.resize(corrected_frame, (1920, 1080))

        corrected_frame = calibrate.my_warp(resized_img)
        # corrected_frame = cv2.warpPerspective(image, matrix2, (1920, 1080))

        cv2.imshow("Frame", frame)
        cv2.namedWindow("Corrected Frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Corrected Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Corrected Frame", corrected_frame)
        cv2.moveWindow("Corrected Frame", 1920, 0)
        
        # cv2.setWindowProperty("Frame", cv2.WND_PROP_TOPMOST, 1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cv2.namedWindow("Corrected Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Corrected Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Corrected Frame", corrected_frame)
cv2.moveWindow("Corrected Frame", 1920, 0)
cv2.waitKey(0)

vs.release()
cv2.destroyAllWindows()
