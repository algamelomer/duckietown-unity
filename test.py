import zmq
import json
import time # Import the time module

context = zmq.Context()
socket = context.socket(zmq.REP) 
socket.bind("tcp://*:5555")

print("Python server started. Waiting for Unity...")

# Get the starting time of the script
start_time = time.time()

while True:
    try:
        # Wait for the image data from Unity (we don't need to use it for this test)
        image_bytes = socket.recv()
        
        # --- Decide on the command based on elapsed time ---
        elapsed_time = time.time() - start_time

        # Check if 3 seconds have passed
        if elapsed_time > 15:
            print("3 seconds passed. Steering right.")
            commands = {
                "throttle": 0.4,  # Go forward
                "steer": 100      # Steer right (use -0.6 for left)
            }
        else:
            print("Going straight...")
            commands = {
                "throttle": 0.5,  # Go forward
                "steer": 0.0      # Go straight
            }
        
    except Exception as e:
        # If any error happens, create a "stop" command
        print(f"An error occurred: {e}")
        commands = {"throttle": 0.0, "steer": 0.0}

    finally:
        # Send the chosen command back to Unity
        json_message = json.dumps(commands)
        socket.send_string(json_message)