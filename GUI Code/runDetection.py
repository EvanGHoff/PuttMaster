# import packages
from collections import deque
import numpy as np
import cv2
import pickle
import threading
import requests
import calibrate
import Add_score
import green2screen
import sendData
import time
import csv
from scipy import integrate
from scipy.spatial.transform import Rotation as R
from threading import Thread, Event


# Assumes ball and hole are separated on the x axis
def findMissDir(hole_center, ball_start, ball_end):
    x_diff = hole_center[0] - ball_end[0]
    y_diff = hole_center[1] - ball_end[1]
    if ball_start[0] > hole_center[0]: # Ball starts on the right
        return "Short" if abs(x_diff) > abs(y_diff) and x_diff < 0 else ("Left" if y_diff < 0 else "Right")
    else: # Ball starts on the left
        return "Short" if abs(x_diff) > abs(y_diff) and x_diff > 0 else ("Left" if y_diff > 0 else "Right")

def sensor_reader(esp32_url, sensor_data_buffer, stop_event):
    while not stop_event.is_set():
        try:
            response = requests.get(esp32_url, timeout=2)
            if response.status_code == 200:
                raw_data = response.text.strip()
                try:
                    accel_line, gyro_line = raw_data.split('\n')
                    accel_vals = {k.strip(): float(v.strip()) for k, v in 
                                  (item.split(':') for item in accel_line.split('|'))}
                    gyro_vals = {k.strip(): float(v.strip().replace('dps', '')) for k, v in 
                                 (item.split(':') for item in gyro_line.split('|'))}

                    # Convert raw units
                    ax = accel_vals['Accel X'] / 256 * 9.8
                    ay = accel_vals['Accel Y'] / 256 * 9.8
                    az = accel_vals['Accel Z'] / 256 * 9.8
                    gx = gyro_vals['Gyro X']

                    timestamp = time.time()
                    sensor_data_buffer.append((timestamp, ax, ay, az, gx))
                except Exception as e:
                    print(f"[ERROR] Parse failed: {e} | Raw: {raw_data}")
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")


def data_logger(sensor_data_buffer, stop_event):
    with open("sensor_log.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'ax', 'ay', 'az', 'gx'])
        while not stop_event.is_set():
            if sensor_data_buffer:
                row = sensor_data_buffer.popleft()
                writer.writerow(row)
                f.flush()

def process_logged_data():
    print("\n[INFO] Processing logged data...")
    timestamps, ax, ay, az, gx = [], [], [], [], []

    with open("sensor_log.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row['timestamp']))
            ax.append(float(row['ax']))
            ay.append(float(row['ay']))
            az.append(float(row['az']))
            gx.append(float(row['gx']))

    ax = np.array(ax)
    ay = np.array(ay)
    az = np.array(az)
    gx = np.array(gx)
    timestamps = np.array(timestamps)

    dts = np.diff(timestamps, prepend=timestamps[0])
    mean_dt = np.mean(dts)

    # Use first second of data as baseline offset (stationary)
    N_offset = int(1.0 / mean_dt)
    ax0, ay0, az0 = np.mean(ax[:N_offset]), np.mean(ay[:N_offset]), np.mean(az[:N_offset])
    gx0 = np.mean(gx[:N_offset])

    # Remove offset
    ax -= ax0
    ay -= ay0
    az -= az0
    gx -= gx0

    # Integrate gyro to get pitch (degrees)
    pitch = integrate.cumulative_trapezoid(gx, dx=mean_dt, initial=0)

    # Rotate acceleration vector to remove gravity
    ax_nog = []
    for i in range(len(ax)):
        rot = R.from_rotvec([-np.deg2rad(pitch[i]), 0, 0])  # only pitch rotation
        g = rot.apply([ax[i], ay[i], az[i]])
        ax_nog.append(g[0])  # horizontal X-direction acceleration

    ax_nog = np.array(ax_nog)

    # Integrate to get velocity (in X)
    vx = integrate.cumulative_trapezoid(ax_nog, dx=mean_dt, initial=0)

    print(f"[RESULT] Final velocity: {vx[-1]:.2f} m/s")
    print(f"[RESULT] Final facing angle (pitch): {pitch[-1]:.2f} degrees")
    return abs(round(vx[-1], 2)), round(pitch[-1], 2)




def cameraDetection(vs, displayFrame, displayCorrected, displayBallMask, displayHoleMask, sensorsOn, user, stop_code):
    prev_time = time.time()
    fps_queue = deque(maxlen=50)  # Store the last 50 FPS values


    if sensorsOn:
        # Replace with your ESP32's IP address
        esp32_url = "http://192.168.220.91"

        # Shared buffer to store last 10 sensor readings
        sensor_data_buffer = deque(maxlen=10)

        stop_event = Event()
        reader_thread = Thread(target=sensor_reader, args=(esp32_url, sensor_data_buffer, stop_event))
        logger_thread = Thread(target=data_logger, args=(sensor_data_buffer, stop_event))
        reader_thread.start()
        logger_thread.start()


    # define the lower and upper boundaries for the ball and hole
    #ball_lower = (80, 25, 130)  # White
    #ball_upper = (120, 75, 255)  # White
    ball_lower = (20, 16, 105)  # Yellow Ball on mat in lab at max light
    ball_upper = (65, 235, 255)  # Yellow
    #ball_lower = (0, 180, 150) # Orange
    #ball_upper = (20, 225, 255) # Orange
    hole_lower = (63, 20, 80)      # Black (hole, to be adjusted once the actual hole is constructed) 
    hole_upper = (100, 95, 197) # Black
    #hole_lower = (90, 10, 40)      # Black (hole, to be adjusted once the actual hole is constructed) 
    #hole_upper = (130, 40, 130) # Black

    # Parameter Initialization
    pts = deque(maxlen=1)
    positions = []  # Store (x, y) positions
    ball_detected = False
    hole_detected = False
    ball_moved = False
    #rectangle_detected = False
    ball_out_of_green = False

    score = 0

    start_time = None

    print(f"Requested FPS: 60, Got {vs.get(cv2.CAP_PROP_FPS)}")


    dst_points = pickle.load(open('matrixes/dstPts.p','rb'))
    number_of_frame = 30

    # homog_matrix, camera_points, aspect_ratio = green2screen.green2screen([dst_points[0], dst_points[3], dst_points[1], dst_points[2]])

    #dst_points = None
    #print(camera_points)
    if dst_points is None:
        #input("Camera point is None, ctrl+c to stop program")
        exit()
    else:
        min_values = np.min(np.array(dst_points).astype(int), axis=0)
        max_values = np.max(np.array(dst_points).astype(int), axis=0)
    # input()



    # Main Video Loop
    while not stop_code.is_set():
        start_time = time.time()
        ret, frame = vs.read()

        
        matrix3by4 = cv2.getPerspectiveTransform(dst_points, np.array([[0, 0], [1440-1, 0], [1440-1, 1080-1], [0, 1080-1]], dtype=np.float32))
        frame = cv2.warpPerspective(frame, matrix3by4, (1440, 1080))

        corrected_frame = np.zeros_like(frame)
        
        #calibrate.rectangle()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ball_mask = cv2.inRange(hsv, ball_lower, ball_upper)
        #ball_mask = cv2.erode(ball_mask, None, iterations=2)
        #ball_mask = cv2.dilate(ball_mask, None, iterations=2)
        if displayBallMask:
            cv2.imshow("ball mask", ball_mask)

        hole_mask = cv2.inRange(hsv, hole_lower, hole_upper)
        #hole_mask = cv2.GaussianBlur(hole_mask, (9, 9), 2)
        #hole_mask = cv2.erode(hole_mask, None, iterations=2)
        #hole_mask = cv2.dilate(hole_mask, None, iterations=2)
        if displayHoleMask:
            cv2.imshow("hole_mask", hole_mask)

        #ball_cnt = cv2.findContours(ball_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #ball_cnts = imutils.grab_contours(ball_cnt)

        '''
        if ball_cnts and not ball_detected:
            print("Ball Detected")
            ball_detected = True
            c = max(ball_cnts, key=cv2.contourArea)
            ((ball_x, ball_y), ball_radius) = cv2.minEnclosingCircle(c)
            ball_center = (int(ball_x), int(ball_y))
            positions.append(ball_center)'''
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                    param1=50, param2=30, minRadius=5, maxRadius=30)
        
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
                    
                    mean_val = cv2.mean(hole_mask, mask)
                
                    #print(mean_val, "   ", (x, y, r))
                    if mean_val[0] > 200 and r > max_radius:  # Ensure it's dark and the largest circle
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
            '''
            c = max(ball_cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))'''
            
            if  center is not None and positions[-1] is not None:
                if radius > 0:
                    cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
                    cv2.circle(corrected_frame, center, int(radius), (0, 255, 255), 2)
                    #cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                # Ball starts moving
                if len(positions) > 2:
                    if not ball_moved and np.linalg.norm(np.array(center) - np.array(positions[-2])) / np.linalg.norm(max_values - min_values) > 0.03:
                        print("Ball is hit!")
                        if sensorsOn:
                            stop_event.set()
                            reader_thread.join()
                            logger_thread.join()
                        ball_moved = True

            elif center is None and positions[-1] is not None:
                positions.append(positions[-1]) 

        # Draw the actual trajectory
        if len(positions) > 1:
            for i in range(1, len(positions)):
                if positions[i - 1] is not None and positions[i] is not None:
                    cv2.line(frame, positions[i - 1], positions[i], (0, 0, 255), 10)
                    cv2.line(corrected_frame, positions[i - 1], positions[i], (255, 255, 255), 10)


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

        # case 1, stops on the green
        if len(positions) > number_of_frame and ball_moved:
            distances = np.linalg.norm(np.array(positions[-number_of_frame:]) - np.array(positions[-1]), axis=1)
            #print(np.mean(distances) / np.linalg.norm(max_values - min_values))
            if 0 < np.mean(distances) / np.linalg.norm(max_values - min_values) < 0.01:
                print("Ball Stopped. Subsystem Stopping...")
                break
            # case 2, out of green
            edge_threshold = 20
            near_edge_count = sum(Add_score.is_near_edge(pos, dst_points, edge_threshold) for pos in positions[-number_of_frame:])
            if near_edge_count > 115 and 0 < np.mean(distances) / np.linalg.norm(max_values - min_values) == 0:
                ball_out_of_green = True

        
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
        print(f"FPS of Projector: {avg_fps:.2f}") 

        corrected_frame = calibrate.my_warp(corrected_frame)

        if displayFrame:
            cv2.imshow("Camera View", frame)
        if displayCorrected:
            cv2.imshow("Projector Display", corrected_frame)


        cv2.namedWindow("Corrected Frame", cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow("Corrected Frame", 1920, 0)
        cv2.setWindowProperty("Corrected Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Corrected Frame", corrected_frame)
        

        if start_time is not None:
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time = end_time - start_time
            #print("Time from data to display:", elapsed_time, "seconds")
            # input()
        
        # cv2.setWindowProperty("Frame", cv2.WND_PROP_TOPMOST, 1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    if sensorsOn:
        puttSpeed, puttPitch = process_logged_data()
    else:
        puttSpeed, puttPitch = 0, 0
    sendData.sendData(user, score, puttSpeed, puttPitch, findMissDir(hole_center, positions[0], positions[-1]))

    corrected_frame = calibrate.my_warp(corrected_frame)
    cv2.namedWindow("Corrected Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Corrected Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Corrected Frame", corrected_frame)
    cv2.moveWindow("Corrected Frame", 1920, 0)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


