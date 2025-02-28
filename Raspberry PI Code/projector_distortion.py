import cv2
import numpy as np
import os

print(cv2.__version__)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


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
    Detects the projected ArUco markers from the camera feed and computes the correction matrix.
    """
    cap = cv2.VideoCapture(camera_index)
    parameters = cv2.aruco.DetectorParameters()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if len(ids) >= 4:
                # Sort corners by marker ID to maintain order
                ordered_corners = [corners[np.where(ids == i)[0][0]][0] for i in range(4)]
                src_points = np.array([
                    ordered_corners[0][0],  # top-left
                    ordered_corners[1][1],  # top-right
                    ordered_corners[2][2],  # bottom-right
                    ordered_corners[3][3]   # bottom-left
                ], dtype="float32")

                # width = 1600
                # height = 800

                width = int(np.linalg.norm(ordered_corners[1][1] - ordered_corners[0][0]))  # Top width
                height = int(np.linalg.norm(ordered_corners[2][2] - ordered_corners[0][0]))  # Left height

                dst_points = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype="float32")

                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                break
                
                # corrected = cv2.warpPerspective(frame, matrix, (width, height))
                # cv2.imshow("Corrected Projection", corrected)

                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    print("Perspective Transformation Matrix:\n", matrix)
                #    break

        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(matrix)

    with open("Raspberry PI Code/distortionMatrix.txt", "w") as f:
        f.write(str(matrix))

    return matrix


def rectangle():
    array = np.zeros([1080, 1920, 3], np.uint8)

    path = r'C:\Users\ehoff\PuttMaster-1\src'
 
    # Reading an image in default mode
    image = array

    start_point = (200, 200)
    end_point = (1800, 900)

    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = -1
    window_name = 'Image'
    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    image = cv2.rectangle(image, start_point, end_point, color, thickness)

    # Displaying the image 

    return image


def image_det():
    # os.environ['DISPLAY'] = ':1.0'
    # Create a VideoCapture object
    # cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    # if not cap.isOpened():
    #    print("Cannot open camera")
    #    exit()

    rect = rectangle()
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", rect)
    cv2.moveWindow("window", 1920, 0)
    cv2.waitKey(0)

    detect_projected_rectangle(0)


    '''
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is read correctly, ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    '''


def detect_projected_rectangle(camera_index=0):
    # Open the camera
    cap = cv2.VideoCapture(camera_index)
    print("Press 'q' when the rectangle is detected and you want to capture the correction matrix.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV and create a mask for the Blue color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Optional: Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangle_contour = None
        max_area = 0

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            area = cv2.contourArea(approx)

            if len(approx) == 4 and area > max_area:
                rectangle_contour = approx
                max_area = area

        # If rectangle is found, draw it on the frame
        if rectangle_contour is not None:
            cv2.drawContours(frame, [rectangle_contour], -1, (0, 255, 0), 2)

        # Display the mask and detected rectangle for debugging
        cv2.imshow("Blue Mask", mask)
        cv2.imshow("Detecting Blue Rectangle", frame)

        # Exit and compute transformation matrix on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q') and rectangle_contour is not None:
            rect = rectangle_contour.reshape(4, 2)
            rect = order_points(rect)

            width = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
            height = int(max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])))

            dst_points = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")

            # Compute the perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(rect, dst_points)
            print("Perspective Transformation Matrix:\n", matrix)

            corrected = cv2.warpPerspective(frame, matrix, (width, height))
            cv2.imshow("Corrected Projection", corrected)
            cv2.waitKey(0)
            break

    cap.release()
    cv2.destroyAllWindows()
    return matrix


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

# rectangle()
# image_det()

generate_aruco_markers()
display_markers()
detect_aruco_markers(0)