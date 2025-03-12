# Define ChArUco board parameters
squares_x = 7  # Number of squares along x-axis
squares_y = 4  # Number of squares along y-axis
square_length = 100  # Length of squares in pixels
marker_length = 80  # ArUco marker size relative to square

board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)

def generate_charuco_board():
    """
    Generates a ChArUco board and saves it as an image.
    """
    board_image = board.generateImage((squares_x * square_length, squares_y * square_length), marginSize=20, borderBits=1)
    save_path = r'C:\Users\ehoff\PuttMaster-1\Raspberry PI Code\markers\charuco_board.png'
    cv2.imwrite(save_path, board_image)
    print("ChArUco board generated and saved.")


def display_charuco_board():
    """
    Displays the generated ChArUco board in full-screen mode on the second screen.
    """
    image_path = r'C:\Users\ehoff\PuttMaster-1\Raspberry PI Code\markers\charuco_board.png'
    board_image = cv2.imread(image_path)
    
    cv2.namedWindow("ChArUco Board", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("ChArUco Board", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("ChArUco Board", board_image)
    cv2.moveWindow("ChArUco Board", 1920, 0)  # Move to second screen

    # while True:
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break


def detect_charuco_markers(cap):
    """
    Detects ChArUco markers and computes the perspective transformation matrix.
    Keeps displaying the board until transformation is found.
    """
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    charuco_detector = cv2.aruco.CharucoDetector(board)  # Create CharucoDetector once

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None and len(ids) > 0:
            # Convert corners and ids to proper numpy arrays
            corners = np.array(corners, dtype=np.float32)
            ids = np.array(ids, dtype=np.int32)
            
            # Detect ChArUco corners from detected ArUco markers
            charuco_corners = np.array([], dtype=np.int32)
            charuco_ids = np.array([], dtype=np.int32)
            _, _, charuco_corners, charuco_ids = charuco_detector.detectBoard(frame, charuco_corners, charuco_ids, corners, ids)
            
            if len(charuco_corners) > 0:
                print("Corner length:", len(charuco_corners), "\nID length:", len(charuco_ids))
                
                # Flatten corner and ID arrays
                charuco_corners_flat = []
                charuco_ids_flat = []

                for corners, ids in zip(charuco_corners, charuco_ids):
                    charuco_corners_flat.append(corners.reshape(-1, 2))
                    charuco_ids_flat.append(ids)

                # Convert lists to numpy arrays
                charuco_corners_flat = np.vstack(charuco_corners_flat)
                charuco_ids_flat = np.hstack(charuco_ids_flat)

                # Ensure they have the same length before passing to the drawing function
                if len(charuco_corners_flat) == len(charuco_ids_flat):
                    # Draw detected corners and ids on the frame
                    cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners_flat, charuco_ids_flat)
                    
                    # Assuming the first four corners are the source points for perspective transform
                    src_points = charuco_corners_flat[:4].reshape(4, 2)

                    # Define destination points (where you want to map the source points to)
                    height, width = frame.shape[:2]
                    dst_points = np.array([
                        [0, 0],  # Top-left corner
                        [width - 1, 0],  # Top-right corner
                        [width - 1, height - 1],  # Bottom-right corner
                        [0, height - 1]  # Bottom-left corner
                    ], dtype=np.float32)

                    # Calculate the perspective transform matrix
                    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

                    # Apply the perspective transform to the frame (flattening the image)
                    corrected = cv2.warpPerspective(frame, matrix, (width, height))

                    np.savetxt("C:\\Users\\ehoff\\PuttMaster-1\\Raspberry PI Code\\distortionMatrix.txt", matrix, delimiter=",")
                    print("Perspective Transformation Matrix:\n", matrix)

                    while True:
                        cv2.imshow("Corrected Perspective", corrected)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    break  # Stop once transformation is found
        
        cv2.imshow("ChArUco Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    while True:
        cv2.imshow("Corrected View", corrected)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    return src_points