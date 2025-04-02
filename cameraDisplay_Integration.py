# import packages
from collections import deque
import numpy as np
import cv2
import imutils
import pickle
import time
import calibrate
import Add_score

# start = input("Press Enter to start the subsystem...")

# define the lower and upper boundaries for the ball and hole
ball_lower = (20, 150, 120)  # Yellow
ball_upper = (40, 235, 255)  # Yellow
#ball_lower = (0, 180, 150) # Orange
#ball_upper = (20, 225, 255) # Orange
hole_lower = (85, 100, 10)      # Black (hole, to be adjusted once the actual hole is constructed) 
hole_upper = (120, 180, 110) # Black
#hole_lower = (90, 10, 40)      # Black (hole, to be adjusted once the actual hole is constructed) 
#hole_upper = (130, 40, 130) # Black

green_lower = (60, 150, 20)
green_upper = (95, 255, 150)

pts = deque(maxlen=1)
positions = []  # Store (x, y) positions
stopped_counter = 0  # Counter to check if ball stops

vs = cv2.VideoCapture(0)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

calibrating = True
if calibrating:
    calibrate.main(vs)

input("Calib Done")

# video input handling

#time.sleep(1.0)
vs.set(cv2.CAP_PROP_FPS, 60)
'''
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
'''


'''

'''

ball_detected = False
hole_detected = False
ball_moved = False
rectangle_detected = False

print(f"Requested FPS: 60, Got {vs.get(cv2.CAP_PROP_FPS)}")

src_points = pickle.load(open('Raspberry PI Code/matrixes/srcPts.p','rb'))
dst_points = pickle.load(open('Raspberry PI Code/matrixes/dstPTS.p','rb'))

while True:
    ret, frame = vs.read()

    if not rectangle_detected:  
        #dst_points = calibrate.order_pts(dst_points)
        if dst_points is None:
            break
        else:
            min_values = np.min(dst_points.astype(int), axis=0)
            max_values = np.max(dst_points.astype(int), axis=0)
            rectangle_detected = True
    
    if rectangle_detected:
        frame = frame.copy()[min_values[1]:max_values[1], min_values[0]:max_values[0]]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        corrected_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        calibrate.rectangle()

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

        if ball_cnts and not ball_detected:
            print("Ball Detected")
            ball_detected = True
            c = max(ball_cnts, key=cv2.contourArea)
            ((ball_x, ball_y), ball_radius) = cv2.minEnclosingCircle(c)
            ball_center = (int(ball_x), int(ball_y))
            positions.append(ball_center)
        
        # Detect circles using Hough Circle Transform
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
                #cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.circle(mask, (x, y), r, 255, -1)
                
                mean_val = cv2.mean(gray, mask)
            
                #print(mean_val, "   ", (x, y, r))
                
                if mean_val[0] < 60 and r > max_radius:  # Ensure it's dark and the largest circle
                    max_radius = r
                    largest_circle = (x, y, r)
            #cv2.imshow("gray", gray)
            #cv2.imshow("mask1", mask)
            # Draw the largest detected hole
            #if largest_circle is not None:
            if largest_circle is not None and not hole_detected:
                hole_x, hole_y, hole_radius = largest_circle
                hole_center = (hole_x, hole_y)
                print("Hole Detected")
                hole_detected = True
                #cv2.circle(frame, hole_center, hole_radius, (0, 255, 0), 2)  # Draw detected hole
                '''
                if ball_detected:
                    cv2.line(frame, ball_center, hole_center, (0, 255, 0), 2)
                    cv2.line(corrected_frame, ball_center, hole_center, (0, 255, 0), 2)
                    optimal_trajectory = (ball_center, hole_center)
                    positions.append(ball_center)
                '''

        center = None
        if len(ball_cnts) > 0 and ball_detected and hole_detected: 
            optimal_trajectory = (ball_center, hole_center)
            cv2.line(frame, optimal_trajectory[0], optimal_trajectory[1], (0, 255, 0), 2)
            cv2.line(corrected_frame, optimal_trajectory[0], optimal_trajectory[1], (0, 255, 0), 2)
            cv2.circle(frame, hole_center, int(hole_radius), (255, 0, 0), 2)
            cv2.circle(corrected_frame, hole_center, int(hole_radius), (255, 0, 0), 2)

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
                '''
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
                '''
            positions.append(center)  # Record position
            #print(" ", ball_center, " ", center)

        # Draw the actual trajectory
        if len(positions) > 1:
            for i in range(1, len(positions)):
                if positions[i - 1] is not None and positions[i] is not None:
                    cv2.line(frame, positions[i - 1], positions[i], (0, 0, 255), 2)
                    cv2.line(corrected_frame, positions[i - 1], positions[i], (0, 0, 255), 2)

        pts.appendleft(center)
        #for i in range(1, len(pts)):
        if pts[0] is not None:
            cv2.line(frame, pts[0], hole_center, (0, 255, 255), 2)
            cv2.line(corrected_frame, pts[0], hole_center, (0, 255, 255), 2)

        #calculate score
        score = 100
        Add_score.add_score_to_image(frame, score)

        corrected_frame = cv2.warpPerspective(corrected_frame, matrix, (1280, 720))

        cv2.imshow("Frame", frame)
        cv2.imshow("Corrected Frame", corrected_frame)
        cv2.setWindowProperty("Frame", cv2.WND_PROP_TOPMOST, 1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

vs.release()
#vs.stop()
cv2.destroyAllWindows()

'''
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
'''
# Display Code 

'''

dst_points = pickle.load(open('Raspberry PI Code/matrixes/dstPTS.p','rb'))

dst_points = order_pts(dst_points)

print("src:", src_points)
print("dst:", dst_points)

prev_time = time.time()

fps_queue = deque(maxlen=50)  # Store the last 50 FPS values


# Display Image
# display_item = display_image()

# Line Test
# display_item = display_line((100, 240), (520, 320))

# Changing Line Test
start_x = 0
start_y = 0

while True:
    # Changing Line Test
    #
    if start_x < 570:
        display_item = display_line((100, 240), (start_x, start_y))
        start_x += 2
        start_y += 1
    else:
        start_x = 0
        start_y = 0
    #

    current_time = time.time()
    time_diff = current_time - prev_time
    prev_time = current_time

    if time_diff > 0:
        fps = 1.0 / time_diff
        fps_queue.append(fps)
    else:
        fps_queue.append(0)

    avg_fps = sum(fps_queue) / len(fps_queue)  # Moving average FPS

    # Perspective transform 
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    corrected_frame = cv2.warpPerspective(display_item, matrix, (1280, 720))
    # corrected_frame = cv2.warpPerspective(display_item, matrix, (710, 420))

    # Display Smoothed FPS
    cv2.putText(corrected_frame, f"FPS: {avg_fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.namedWindow("Corrected Frame", cv2.WINDOW_NORMAL )
    cv2.setWindowProperty("Corrected Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Corrected Frame", corrected_frame)
    cv2.moveWindow("Corrected Frame", 1920, 0)  # Move to second screen
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
cv2.destroyAllWindows()

'''