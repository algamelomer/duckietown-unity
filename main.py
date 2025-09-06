import zmq
import json
import numpy as np
import cv2

# --- ZMQ Connection Setup ---
context = zmq.Context()
socket = context.socket(zmq.REP) 
socket.bind("tcp://*:5555")

# Create a Poller to handle non-blocking receives
poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

print("Final Python script started. Waiting for Unity...")

# --- Main Loop ---
while True:
    # Default commands
    throttle = 0.0
    steer = 0.0

    # Poll for incoming messages with a timeout of 1ms
    socks = dict(poller.poll(timeout=1))

    # If the socket has a message, process it
    if socket in socks and socks[socket] == zmq.POLLIN:
        image_bytes = socket.recv()

        # Convert byte array to an image
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        # Display the image if it's valid
        if image_bgr is not None:
            cv2.imshow("Unity Camera View", image_bgr)
        
        # --- Get Keyboard Input ---
        # This part now only runs when we're about to send a reply
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('w'):
            throttle = 1.0
        elif key == ord('s'):
            throttle = -1.0
        elif key == ord('a'):
            throttle = 0.3
            steer = -1.0
        elif key == ord('d'):
            throttle = 0.3
            steer = 1.0
        elif key == ord('q'):
            print("Quitting...")
            break
        
        # --- Send Reply ---
        # We ONLY send a reply after we've received a request
        commands = {"throttle": throttle, "steer": steer}
        json_message = json.dumps(commands)
        socket.send_string(json_message)

    # --- Handle Window and Keys Even If No Message ---
    # We must still call waitKey to keep the window open and responsive
    # and to check for the quit key
    else:
        # Create a dummy frame to show if we're not receiving images yet
        if 'image_bgr' not in locals() or image_bgr is None:
             display_frame = np.zeros((200, 400, 3), dtype=np.uint8)
             cv2.putText(display_frame, "Waiting for video from Unity...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
             cv2.imshow("Unity Camera View", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break

# Clean up
print("Closing resources.")
socket.close()
context.term()
cv2.destroyAllWindows()