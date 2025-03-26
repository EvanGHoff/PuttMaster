# import packages
from collections import deque
import numpy as np
import cv2
import imutils
import os, pdb
import pickle
import time

# start = input("Press Enter to start the subsystem...")

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


# Display Code 


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def detect_green_rectangle():
    """
    Detects a specific green rectangular shape, applies perspective transformation,
    and extracts the corrected shape region.
    """
    cap = cv2.VideoCapture(1)

    shape_corners = None  # Store the detected corners

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            break

        # Convert to HSV and create a mask for green color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, green_lower, green_upper)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            shape_corners = best_corners

            # Draw the detected rectangle
            cv2.drawContours(frame, [best_corners.astype(int)], -1, (0, 255, 0), 3)
            for (x, y) in shape_corners.astype(int):
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Draw red circles at corners

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Return the detected rectangle's corners if found
    if shape_corners is not None and len(shape_corners) == 4:
        return shape_corners
    else:
        print("No valid rectangle detected.")
        return None


def rectangle():
    array = np.zeros([1080, 1920, 3], np.uint8)
 
    # Reading an image in default mode
    image = array

    start_point = (200, 200)
    end_point = (1800, 900)

    color = (0, 0, 0)

    # Line thickness of 2 px
    thickness = -1
    window_name = 'Image'
    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    image = cv2.rectangle(image, start_point, end_point, color, thickness)

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", image)
    cv2.moveWindow("window", 1920, 0)

    # Displaying the image 

    return image


def order_points(pts):
    """
    Orders the 4 points in the order:
    top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference for point ordering
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect


def generate_aruco_markers():
    """
    Generates four ArUco markers with unique IDs and saves them as images.
    """
    marker_size = 200  # Size of each marker in pixels
    save_path = r'C:\Users\\ehoff\\PuttMaster-1\\Raspberry PI Code\\markers'
    os.makedirs(save_path, exist_ok=True)

    for marker_id in range(4):
        marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
        marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size, marker_image, 1)
        cv2.imwrite(os.path.join(save_path, f'marker_{marker_id}.png'), marker_image)
    print("ArUco markers generated and saved.")


def display_markers():
    """
    Displays the generated ArUco markers in a full-screen window.
    """
    markers = []
    for i in range(4):
        marker = cv2.imread(f'C:/Users/ehoff/PuttMaster-1/Raspberry PI Code/markers/marker_{i}.png')
        markers.append(marker)

    # Create a white canvas and place the markers at the corners
    canvas = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    canvas[100:300, 100:300] = markers[0]  # Top-left
    canvas[100:300, 1620:1820] = markers[1]  # Top-right
    canvas[780:980, 1620:1820] = markers[2]  # Bottom-right
    canvas[780:980, 100:300] = markers[3]  # Bottom-left

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", canvas)
    cv2.moveWindow("window", 1920, 0)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def detect_aruco_markers(camera_index=0):
    """
    Detects the projected ArUco markers from the camera feed and returns the detected source points.
    """
    cap = cv2.VideoCapture(camera_index)
    parameters = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)  # Define marker dictionary

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None and len(ids) >= 4:
            # Create a dictionary mapping marker IDs to their corner coordinates
            id_to_corners = {ids[i][0]: corners[i][0] for i in range(len(ids))}

            # Ensure we only use the four lowest IDs detected
            sorted_ids = sorted(id_to_corners.keys())[:4]

            if len(sorted_ids) == 4:
                # Extract corners in a consistent order
                ordered_corners = [id_to_corners[i] for i in sorted_ids]

                src_points = np.array([
                    ordered_corners[0][0],  # top-left
                    ordered_corners[1][1],  # top-right
                    ordered_corners[2][2],  # bottom-right
                    ordered_corners[3][3]   # bottom-left
                ], dtype="float32")

                cap.release()
                cv2.destroyAllWindows()

                

                # Sort by y-coordinates (top 2, bottom 2)
                sorted_by_y = src_points[np.argsort(src_points[:, 1])]

                # Top two points (left & right)
                top_two = sorted_by_y[:2]
                top_left, top_right = sorted(top_two, key=lambda p: p[0])  # Sort by x

                # Bottom two points (left & right)
                bottom_two = sorted_by_y[2:]
                bottom_left, bottom_right = sorted(bottom_two, key=lambda p: p[0])  # Sort by x

                # Reorder correctly for getPerspectiveTransform
                # src_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

                return src_points  # Return detected source points

        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None  # Return None if detection fails


def order_pts(point_list):
    sum_list = []
    index_list = [0, 1, 2, 3]
    biggest_index = 0  # index of top right
    smallest_index = 0  # index of bottom left
    for i in range(len(point_list)):
        sum_list.append(point_list[i][0] + point_list[i][1])
        if sum_list[i] > sum_list[biggest_index]:
            biggest_index = i
        if sum_list[i] < sum_list[smallest_index]:
            smallest_index = i

    top_left = smallest_index
    btm_right = biggest_index

    index_list.remove(top_left)
    index_list.remove(btm_right)
    
    if point_list[index_list[0]][0] > point_list[index_list[1]][0]:  # bigger x
        top_right = index_list[0]
        btm_left = index_list[1]
    else:
        top_right = index_list[1]
        btm_left = index_list[0]
    
    ordered_list = [point_list[top_left], point_list[top_right], point_list[btm_right], point_list[btm_left]]

    return np.array(ordered_list, dtype=np.float32)


def display_image():
    image = cv2.imread(r"C:\Users\ehoff\PuttMaster-1\Raspberry PI Code\blue.jpg")
    if image is None:
        print("Error: Could not load image.")
        exit()

    # Calculate width and height of the destination rectangle
    src_width = int(np.linalg.norm(src_points[1] - src_points[0]))  # Top width
    src_height = int(np.linalg.norm(src_points[3] - src_points[0]))  # Left height

    # Calculate width and height of the destination rectangle
    dst_width = int(np.linalg.norm(dst_points[1] - dst_points[0]))  # Top width
    dst_height = int(np.linalg.norm(dst_points[3] - dst_points[0]))  # Left height

    # Calculate scaling factors for width and height
    scale_x = dst_width / src_width
    scale_y = dst_height / src_height

    scaled_image = cv2.resize(image, None, fx=.7, fy=1, interpolation=cv2.INTER_LINEAR)

    return scaled_image


def display_image_3m():
    image = cv2.imread(r"C:\Users\ehoff\PuttMaster-1\Raspberry PI Code\blue.jpg")
    if image is None:
        print("Error: Could not load image.")
        exit()

    # Calculate width and height of the destination rectangle
    src_width = int(np.linalg.norm(src_points[1] - src_points[0]))  # Top width
    src_height = int(np.linalg.norm(src_points[3] - src_points[0]))  # Left height

    # Calculate width and height of the destination rectangle
    dst_width = int(np.linalg.norm(dst_points[1] - dst_points[0]))  # Top width
    dst_height = int(np.linalg.norm(dst_points[3] - dst_points[0]))  # Left height

    # Calculate scaling factors for width and height
    scale_x = dst_width / src_width
    scale_y = dst_height / src_height

    scaled_image = cv2.resize(image, None, fx=.75, fy=1, interpolation=cv2.INTER_LINEAR)

    return scaled_image


def display_line(start_point, end_point):
    # Define the original screen size before warping (same as input image size)
    original_width, original_height = 640, 480 
    
    # Create a blank image (white background for visibility)
    base_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    # Draw the line on the base image
    line_color = (255, 255, 255)  # Red color
    line_thickness = 3
    cv2.line(base_image, start_point, end_point, line_color, line_thickness)

    return base_image


calibrating = False

if calibrating:
    # src
    display_markers()
    src_points = detect_aruco_markers(1)
    #pickle.dump(src_points, open('Raspberry PI Code/matrixes/srcPts.p','wb'))

    # dst
    rectangle()
    cv2.waitKey(1)
    dst_points = detect_green_rectangle()
    #pickle.dump(dst_points, open('Raspberry PI Code/matrixes/dstPts.p','wb'))
    input() #pause to find points


src_points = pickle.load(open('Raspberry PI Code/matrixes/srcPts.p','rb'))
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
    #'''
    if start_x < 570:
        display_item = display_line((100, 240), (start_x, start_y))
        start_x += 2
        start_y += 1
    else:
        start_x = 0
        start_y = 0
    #'''

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
    corrected_frame = cv2.warpPerspective(display_item, matrix, (640, 480))
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