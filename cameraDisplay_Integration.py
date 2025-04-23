# import packages
from collections import deque
import numpy as np
import cv2
import pickle
import calibrate
import Add_score
import time

prev_time = time.time()

fps_queue = deque(maxlen=10)  # Store the last 50 FPS values

# define the lower and upper boundaries for the ball and hole
#ball_lower = (20, 150, 120)  # Yellow
#ball_upper = (40, 235, 255)  # Yellow
ball_lower = (30, 90, 180)  # Yellow Ball on mat in lab at max light
ball_upper = (65, 235, 255)  # Yellow
#ball_lower = (0, 180, 150) # Orange
#ball_upper = (20, 225, 255) # Orange
hole_lower = (85, 73, 35)      # Black (hole, to be adjusted once the actual hole is constructed) 
hole_upper = (120, 160, 160) # Black
#hole_lower = (90, 10, 40)      # Black (hole, to be adjusted once the actual hole is constructed) 
#hole_upper = (130, 40, 130) # Black

# Parameter Initialization
pts = deque(maxlen=1)
positions = []  # Store (x, y) positions
ball_detected = False
hole_detected = False
ball_moved = False
rectangle_detected = False
ball_out_of_green = False

score = 0

start_time = None

# Camera Initialization and parameter adjustment 
vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vs.set(cv2.CAP_PROP_FPS, 60)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Calibration
calibrating = False
if calibrating:
    calibrate.main(vs)
print(f"Requested FPS: 60, Got {vs.get(cv2.CAP_PROP_FPS)}")


dst_points = pickle.load(open('Raspberry PI Code/matrixes/dstPts.p','rb'))

# Main Video Loop
while True:
    ret, frame = vs.read()

    if not rectangle_detected:  
        if dst_points is None:
            break
        else:
            min_values = np.min(dst_points.astype(int), axis=0)
            max_values = np.max(dst_points.astype(int), axis=0)
            print("Min:", min_values)
            print("Max:", max_values)
            rectangle_detected = True
    
    if rectangle_detected:
        frame = frame[min_values[1]:max_values[1], min_values[0]:max_values[0]]
        corrected_frame = np.zeros((max_values[1] - min_values[1], max_values[0] - min_values[0], 3), dtype=np.uint8)

        #calibrate.rectangle()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ball_mask = cv2.inRange(hsv, ball_lower, ball_upper)
        #ball_mask = cv2.erode(ball_mask, None, iterations=2)
        #ball_mask = cv2.dilate(ball_mask, None, iterations=2)
        cv2.imshow("ball mask", ball_mask)

        hole_mask = cv2.inRange(hsv, hole_lower, hole_upper)
        #hole_mask = cv2.GaussianBlur(hole_mask, (9, 9), 2)
        #hole_mask = cv2.erode(hole_mask, None, iterations=2)
        #hole_mask = cv2.dilate(hole_mask, None, iterations=2)
        cv2.imshow("hole_mask", hole_mask)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                    param1=50, param2=30, minRadius=5, maxRadius=100)
        
        if not ball_detected:
            largest_circle = None
            max_radius = 0
            if circles is not None:
                circles1 = np.round(circles[0, :]).astype("int")  # Convert to integer values
                
                for (x, y, r) in circles1:
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                    mask = np.zeros(frame.shape[:2], dtype="uint8")
                    cv2.circle(mask, (x, y), r, 255, -1)
                    
                    mean_val = cv2.mean(ball_mask, mask)
                    #print(mean_val, "   ", (x, y, r))
                    if mean_val[0] > 200 and r > max_radius:
                        max_radius = r
                        largest_circle = (x, y, r)
                if largest_circle is not None:
                    ball_x, ball_y, ball_radius = largest_circle
                    ball_center = (ball_x, ball_y)
                    ball_detected = True
                    print("Ball Detected")
                    cv2.circle(frame, ball_center, int(ball_radius), (255, 0, 0), 2)
                    cv2.circle(corrected_frame, ball_center, int(ball_radius), (255, 0, 0), 2)
                    positions.append(ball_center)
        
        # Detect circles using Hough Circle Transform
        if not hole_detected:
            # Find the largest detected circle
            largest_circle = None
            max_radius = 0

            if circles is not None:
                circles2 = np.round(circles[0, :]).astype("int")  # Convert to integer values
                
                for (x, y, r) in circles2:
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
            cv2.line(frame, optimal_trajectory[0], optimal_trajectory[1], (0, 255, 0), 10)
            cv2.line(corrected_frame, optimal_trajectory[0], optimal_trajectory[1], (0, 255, 0), 10)
            cv2.circle(frame, hole_center, int(hole_radius), (255, 0, 0), 2)
            cv2.circle(corrected_frame, hole_center, int(hole_radius), (255, 0, 0), 2)

            if circles is not None:
                largest_circle = None
                max_radius = 0
                circles3 = np.round(circles[0, :]).astype("int")  # Convert to integer values
                
                for (x, y, r) in circles3:
                    #cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                    mask = np.zeros(frame.shape[:2], dtype="uint8")
                    cv2.circle(mask, (x, y), r, 255, -1)
                    
                    mean_val = cv2.mean(ball_mask, mask)
                
                    #print(mean_val, "   ", (x, y, r))
                    if mean_val[0] > 200 and r > max_radius:  # Ensure it's dark and the largest circle
                        max_radius = r
                        largest_circle = (x, y, r)
                if largest_circle is not None:
                    ball_x, ball_y, radius = largest_circle
                    center = (ball_x, ball_y)
                    #cv2.circle(frame, ball_center, int(radius), (255, 0, 0), 2)
                    #cv2.circle(corrected_frame, ball_center, int(radius), (255, 0, 0), 2)
                    positions.append(center)
            
            if center is not None and positions[-1] is not None:
                if radius > 0:
                    cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
                    cv2.circle(corrected_frame, center, int(radius), (0, 255, 255), 2)
                    #cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                # Ball starts moving
                #print(np.linalg.norm(np.array(center) - np.array(positions[-2])) / np.linalg.norm(max_values - min_values))
                #print(np.linalg.norm(np.array(center) - np.array(positions[-2])))
                #print(center, positions[-2])
                if not ball_moved and np.linalg.norm(np.array(center) - np.array(positions[-2])) / np.linalg.norm(max_values - min_values) > 0.03:
                    print("Ball is hit!")
                    start_time = time.time()
                    ball_moved = True

            elif center is None and positions[-1] is not None:
                positions.append(positions[-1]) 

        # Draw the actual trajectory
        if len(positions) > 1:
            for i in range(1, len(positions)):
                if positions[i - 1] is not None and positions[i] is not None:
                    cv2.line(frame, positions[i - 1], positions[i], (0, 0, 255), 10)
                    cv2.line(corrected_frame, positions[i - 1], positions[i], (255, 255, 255), 10)

        # case 1, stops on the green
        if len(positions) > 120 and ball_moved:
            distances = np.linalg.norm(np.array(positions[-120:]) - np.array(positions[-1]), axis=1)
            print(np.mean(distances) / np.linalg.norm(max_values - min_values))
            if 0 < np.mean(distances) / np.linalg.norm(max_values - min_values) < 0.01:
                print("Ball Stopped. Subsystem Stopping...")
                break
            # case 2, out of green
            edge_threshold = 20
            near_edge_count = sum(Add_score.is_near_edge(pos, dst_points, edge_threshold) for pos in positions[-120:])
            if near_edge_count > 115 and 0 < np.mean(distances) / np.linalg.norm(max_values - min_values) == 0:
                ball_out_of_green = True

        pts.appendleft(center)
        if pts[0] is not None:
            cv2.line(frame, pts[0], hole_center, (0, 255, 255), 10)
            cv2.line(corrected_frame, pts[0], hole_center, (0, 255, 255), 10)

            if ball_out_of_green:
                Add_score.add_score_to_image(corrected_frame, score)
                break
            else:
                score = Add_score.get_Score(np.array(pts[0]), np.array(hole_center), dst_points, score)
                Add_score.add_score_to_image(corrected_frame, score)
        else:
            Add_score.add_score_to_image(corrected_frame, score)


        current_time = time.time()
        time_diff = current_time - prev_time
        prev_time = current_time

        if time_diff > 0:
            fps = 1.0 / time_diff
            fps_queue.append(fps)
        else:
            fps_queue.append(0)

        avg_fps = sum(fps_queue) / len(fps_queue)  # Moving average FPS

        # Display Smoothed FPS
        # print(f"FPS of Projector: {avg_fps:.2f}")
        
        resized_img = cv2.resize(corrected_frame, (1920, 1080))

        corrected_frame = calibrate.my_warp(resized_img)
        # corrected_frame = cv2.warpPerspective(image, matrix2, (1920, 1080))

        cv2.imshow("Frame", frame)
        cv2.namedWindow("Corrected Frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Corrected Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Corrected Frame", corrected_frame)
        cv2.moveWindow("Corrected Frame", 1920, 0)

        if start_time is not None:
            end_time = time.time()

            # Calculate elapsed time
            elapsed_time = end_time - start_time
            # print("Time from data to display:", elapsed_time, "seconds")
            # input()
        
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
