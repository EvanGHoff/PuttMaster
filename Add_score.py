import cv2
import numpy as np

def add_score_to_image(image: np.ndarray, score: float) -> np.ndarray:
    # Convert to grayscale and create a binary mask of non-empty regions
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
    
    h, w = mask.shape
    block_size = 40  # Minimum space needed for the text
    best_x, best_y = None, None
    
    # Search for an empty region using a grid-based approach
    for y in range(0, h - block_size, 10):  # Step size of 10 pixels
        for x in range(0, w - block_size, 10):
            roi = mask[y:y+block_size, x:x+block_size]
            if np.sum(roi) == 0:  # Completely empty space found
                best_x, best_y = x, y
                break
        if best_x is not None:
            break
    
    # If a valid spot is found, draw the text
    if best_x is not None and best_y is not None:
        text = f"Score: {score}"
        cv2.putText(image, text, (best_x, best_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        best_x = 0
        best_y = 0
        text = f"Score: {score}"
        cv2.putText(image, text, (best_x, best_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)
        #print("Warning: No empty space found for text placement.")

    
    return image

def get_Score(ball_pos, hole_pos, green_corners, curr_score):
    dist = int(np.linalg.norm(ball_pos - hole_pos))

    green_corners = green_corners - green_corners[0]

    gr_dist1 = np.linalg.norm(green_corners[0] - hole_pos)
    gr_dist2 = np.linalg.norm(green_corners[1] - hole_pos)
    gr_dist3 = np.linalg.norm(green_corners[2] - hole_pos)
    gr_dist4 = np.linalg.norm(green_corners[3] - hole_pos)

    max_dist = max(gr_dist1, gr_dist2, gr_dist3, gr_dist4)
 
    return max(curr_score, int(100 - (100 * (dist / max_dist))))

def is_near_edge(ball_pos, green_corners, edge_threshold):
    #ball_x, ball_y = ball_pos
    rect_bounds = green_corners - green_corners[0]
    x_min, x_max, y_min, y_max = rect_bounds
    #print(ball_x - x_min)
    
    near_left   = np.linalg.norm(ball_pos - x_min) <= edge_threshold
    near_right  = np.linalg.norm(x_max - ball_pos) <= edge_threshold
    near_top    = np.linalg.norm(ball_pos - y_min) <= edge_threshold
    near_bottom = np.linalg.norm(y_max - ball_pos) <= edge_threshold

    return near_left or near_right or near_top or near_bottom
