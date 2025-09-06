import zmq
import json
import numpy as np
import cv2 # OpenCV

# --- ZMQ Connection Setup ---
context = zmq.Context()
socket = context.socket(zmq.REP) 
socket.bind("tcp://*:5555")
print("Python server started. Waiting for Unity to connect...")

def process_image_for_lanes(image_bgr):
    """
    Lane following algorithm using Canny edge detection and Hough Lines.
    """
    img_h, img_w, _ = image_bgr.shape
    
    # 1. Region of Interest (ROI)
    roi_vertices = np.array([
        [(0, img_h), (img_w / 2 - 40, img_h / 2), (img_w / 2 + 40, img_h / 2), (img_w, img_h)]
    ], dtype=np.int32)
    mask_roi = np.zeros_like(image_bgr)
    cv2.fillPoly(mask_roi, roi_vertices, (255, 255, 255))
    masked_image = cv2.bitwise_and(image_bgr, mask_roi)

    # 2. Color filtering
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([255, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
    
    # 3. Edge detection
    edges = cv2.Canny(combined_mask, 50, 150)

    # 4. Find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, np.array([]), minLineLength=15, maxLineGap=10)
    
    steer = 0.0
    throttle = 0.4

    if lines is not None:
        lane_center_x = 0
        line_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lane_center_x += (x1 + x2) / 2
            line_count += 1
            cv2.line(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if line_count > 0:
            avg_lane_center = lane_center_x / line_count
            error = avg_lane_center - (img_w / 2)
            steer = -0.8 * (error / (img_w / 2))
            steer = np.clip(steer, -1.0, 1.0)
    else:
        throttle = 0.2
            
    cv2.imshow("Lane Following View", image_bgr)
    cv2.waitKey(1)
            
    return throttle, steer

# --- Main Loop ---
while True:
    commands = {"throttle": 0.0, "steer": 0.0} # Default stop command
    try:
        # Receive image bytes from Unity
        image_bytes = socket.recv()
        
        # Convert byte array to a NumPy array for OpenCV
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        if image_bgr is not None:
            # Process the image to get driving commands
            throttle_command, steer_command = process_image_for_lanes(image_bgr)
            commands = {"throttle": float(throttle_command), "steer": float(steer_command)}
        
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # This part is GUARANTEED to run, ensuring Unity always gets a reply
        json_message = json.dumps(commands)
        socket.send_string(json_message)