# debug_control.py
import zmq
import json
import numpy as np
import cv2
import time
from aruco_library import ArucoType, detect_and_draw_markers

# --- ZMQ Connection Setup ---
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

print("Python lane following script started. Waiting for Unity...")

# ===================================================================
#                      CONTROLS & PARAMETERS
# ===================================================================
PARK_ON_DETECT = True
PARKING_TAG_ID = 0
PARKING_DISTANCE_THRESHOLD = 0.4 

camera_matrix = np.array([
    [1432.0, 0.0,    983.0], 
    [0.0,    1434.0, 561.0], 
    [0.0,    0.0,    1.0]
])  
dist_coeffs = np.array([0.05994318, -0.26432366, -0.00135378, -0.00081574, 0.29707202])

intersection_state = 0
action_start_time = 0
parking_direction = 0 

def is_parking_marker_visible(ids, tvecs):
    if ids is not None:
        for i, marker_id in enumerate(ids):
            distance = tvecs[i][2][0]
            if marker_id[0] == PARKING_TAG_ID and distance < PARKING_DISTANCE_THRESHOLD:
                return True
    return False

def process_image_for_lanes(image_bgr):
    global intersection_state, action_start_time, parking_direction
    
    img_h, img_w, _ = image_bgr.shape
    debug_image = np.copy(image_bgr)

    # 1. ALWAYS GET SENSOR DATA (MARKERS AND LANES)
    debug_image, ids, tvecs, corners = detect_and_draw_markers(
        debug_image, ArucoType.DICT_6X6_250, camera_matrix, dist_coeffs
    )
    
    roi_vertices = np.array([[(0, img_h), (img_w / 2 - 40, img_h / 2), (img_w / 2 + 40, img_h / 2), (img_w, img_h)]], dtype=np.int32)
    mask_roi = np.zeros_like(image_bgr); cv2.fillPoly(mask_roi, roi_vertices, (255, 255, 255))
    masked_image = cv2.bitwise_and(image_bgr, mask_roi)
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100]); upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    lower_white = np.array([0, 0, 180]); upper_white = np.array([255, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    yellow_edges = cv2.Canny(yellow_mask, 50, 150); white_edges = cv2.Canny(white_mask, 50, 150)
    yellow_lines = cv2.HoughLinesP(yellow_edges, 1, np.pi / 180, 20, np.array([]), minLineLength=15, maxLineGap=10)
    white_lines = cv2.HoughLinesP(white_edges, 1, np.pi / 180, 20, np.array([]), minLineLength=15, maxLineGap=10)
    left_count = 0 if yellow_lines is None else len(yellow_lines)
    right_count = 0 if white_lines is None else len(white_lines)

    # 2. HANDLE ONGOING ACTIONS
    if intersection_state != 0:
        elapsed_time = time.time() - action_start_time
        if intersection_state == 4 and elapsed_time > 3.5:
            intersection_state = 0
        elif intersection_state == 3 and elapsed_time > 1.0:
            intersection_state = 0
            
    # 3. MAKE NEW DECISIONS IF IN NORMAL STATE
    if intersection_state == 0:
        gap_on_right = right_count == 0 and left_count > 0
        gap_on_left = left_count == 0 and right_count > 0

        if (gap_on_right or gap_on_left) and is_parking_marker_visible(ids, tvecs) and PARK_ON_DETECT:
            parking_direction = -1 if gap_on_right else 1
            intersection_state = 4
            action_start_time = time.time()
            print(f"Gap detected with parking marker. Entering parking.")
        
        elif left_count == 0 and right_count == 0:
            intersection_state = 3; action_start_time = time.time()

        elif left_count > 0 and right_count > 0:
            # Draw detected lane lines on the debug image
            for line in white_lines: cv2.line(debug_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 0, 255), 2)
            for line in yellow_lines: cv2.line(debug_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 255), 2)

            left_x_sum = sum([(l[0][0] + l[0][2]) / 2 for l in yellow_lines])
            right_x_sum = sum([(l[0][0] + l[0][2]) / 2 for l in white_lines])
            avg_left_x = left_x_sum / left_count; avg_right_x = right_x_sum / right_count
            lane_center_x = (avg_left_x + avg_right_x) / 2
            error = lane_center_x - (img_w / 2)
            steer = -0.8 * (error / (img_w / 2)); steer = np.clip(steer, -1.0, 1.0)
            cv2.line(debug_image, (int(lane_center_x), img_h), (int(img_w/2), int(img_h/2)), (0, 255, 0), 3)

    # 4. EXECUTE ACTIONS BASED ON CURRENT STATE
    steer = 0.0; throttle = 0.4; status_text = "Following Lane"
    if intersection_state == 4:
        status_text = "Entering Parking"; steer = 0.5 * parking_direction; throttle = 0.25
    elif intersection_state == 3:
        status_text = "Stopping at Intersection"; steer = 0.0; throttle = 0.0

    # 5. DRAW INFO & RETURN
    cv2.putText(debug_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(debug_image, f"Steer: {steer:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(debug_image, f"Throttle: {throttle:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # <--- NEW: VISUAL DEBUGGING DASHBOARD ---
    # Create the 2x2 grid for visualization
    h, w, _ = debug_image.shape
    h_small, w_small = h // 2, w // 2

    # Resize components
    main_view = cv2.resize(debug_image, (w_small, h_small))
    roi_view = cv2.resize(masked_image, (w_small, h_small))
    
    # Convert single-channel masks to 3-channel BGR to stack them
    yellow_mask_bgr = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
    white_mask_bgr = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
    
    yellow_view = cv2.resize(yellow_mask_bgr, (w_small, h_small))
    white_view = cv2.resize(white_mask_bgr, (w_small, h_small))

    # Add text labels to each view
    cv2.putText(main_view, "Main View", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(roi_view, "Region of Interest", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(yellow_view, "Yellow Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(white_view, "White Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Stack the images into a dashboard
    top_row = np.hstack((main_view, roi_view))
    bottom_row = np.hstack((yellow_view, white_view))
    dashboard = np.vstack((top_row, bottom_row))

    cv2.imshow("Self-Driving Debug Dashboard", dashboard)
    # <--- End of new section ---

    return throttle, steer

# --- Main Loop (Unchanged) ---
while True:
    socks = dict(poller.poll(timeout=1))
    if socket in socks and socks[socket] == zmq.POLLIN:
        image_bytes = socket.recv()
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image_bgr is not None:
            throttle, steer = process_image_for_lanes(image_bgr)
            commands = {"throttle": float(throttle), "steer": -float(steer)}
            socket.send_string(json.dumps(commands))
        else:
            socket.send_string(json.dumps({"throttle": 0.0, "steer": 0.0}))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting..."); break
# --- Clean up (Unchanged) ---
print("Closing resources.")
socket.close()
context.term()
cv2.destroyAllWindows()