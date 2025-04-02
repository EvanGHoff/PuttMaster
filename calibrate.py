import numpy as np
import cv2
import pickle

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

def display_markers():
    """
    Displays the generated ArUco markers in a full-screen window.
    """
    markers = []
    for i in range(4):
        marker = cv2.imread(f'Raspberry PI Code/markers/marker_{i}.png')
        markers.append(marker)

    # Create a white canvas and place the markers at the corners
    canvas = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    canvas[20:220, 20:220] = markers[0]  # Top-left
    canvas[20:220, 1700:1900] = markers[1]  # Top-right
    canvas[860:1060, 1700:1900] = markers[2]  # Bottom-right
    canvas[860:1060, 20:220] = markers[3]  # Bottom-left

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", canvas)
    cv2.moveWindow("window", 1920, 0)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def detect_aruco_markers(cap):
    """
    Detects the projected ArUco markers from the camera feed and returns the detected source points.
    """
    
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

            # Show the video with detected markers
            cv2.imshow("ArUco Marker Detection", frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if len(sorted_ids) == 4:
                # Extract corners in a consistent order
                ordered_corners = [id_to_corners[i] for i in sorted_ids]

                #print(ordered_corners)
                #input()

                src_points = np.array([
                    ordered_corners[0][0],  # top-left
                    ordered_corners[1][1],  # top-right
                    ordered_corners[2][2],  # bottom-right
                    ordered_corners[3][3]   # bottom-left
                ], dtype="float32")


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

    cv2.destroyAllWindows()
    return None  # Return None if detection fails

def detect_green_rectangle(cap):
    """
    Detects a specific green rectangular shape, applies perspective transformation,
    and extracts the corrected shape region.
    """

    shape_corners = None  # Store the detected corners

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            break

        # Convert to HSV and create a mask for green color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #green_lower = (60, 150, 20)
        #green_upper = (95, 255, 150)

        green_lower = (45, 80, 45)
        green_upper = (102, 210, 185)
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
    

def main(cap):

    # cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # src
    display_markers()
    src_points = detect_aruco_markers(cap)
    print("Markers Detected:", src_points)
    cv2.destroyAllWindows()
    pickle.dump(src_points, open('Raspberry PI Code/matrixes/srcPts.p','wb'))

    # dst
    rectangle()
    cv2.waitKey(1)
    dst_points = detect_green_rectangle(cap)
    dst_points = order_pts(dst_points)
    print("Green Detected:", dst_points)
    pickle.dump(dst_points, open('Raspberry PI Code/matrixes/dstPts.p','wb'))
    #input()  pause to find points

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    pickle.dump(matrix, open('Raspberry PI Code/matrixes/matrix.p','wb'))

if __name__ == "__main__":
    main()