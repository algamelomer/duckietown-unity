import zmq
import json
import numpy as np
import cv2
import tensorflow as tf

print("Setting UP")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- ZMQ Connection Setup ---
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# Create a Poller to handle non-blocking receives
poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

print("Python script started. Waiting for ZMQ client...")

# Define max throttle for slower speed
max_throttle = 0.3

def preProcess(img):
    """
    Pre-processes the entire image for the TFLite model's input format.
    It no longer crops the image.
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

def process_image_and_predict(image_bgr):
    """
    Uses the TFLite model to predict steering and returns a fixed throttle.
    """
    # Pre-process image for model prediction
    processed_image = preProcess(image_bgr)
    image_input = np.array([processed_image], dtype=np.float32)

    # Set the tensor to the input data and run the interpreter
    interpreter.set_tensor(input_details[0]['index'], image_input)
    interpreter.invoke()

    # Get the prediction for steering only
    predictions = interpreter.get_tensor(output_details[0]['index'])
    steer = float(predictions[0][0])

    # Use the predefined throttle value
    throttle = max_throttle

    # Display the processed image for debugging
    processed_display = cv2.cvtColor((processed_image * 255).astype(np.uint8), cv2.COLOR_YUV2BGR)
    cv2.imshow("AI's View", processed_display)

    return throttle, steer

if __name__ == '__main__':
    # Load TFLite model and allocate tensors
    try:
        interpreter = tf.lite.Interpreter(model_path='model.tflite')
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("Model Input Shape:", input_details[0]['shape'])
        print("Model Output Shape:", output_details[0]['shape'])

    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        exit()

    # Main ZMQ loop
    while True:
        socks = dict(poller.poll(timeout=1))

        if socket in socks and socks[socket] == zmq.POLLIN:
            try:
                # Receive the image bytes from the ZMQ client
                image_bytes = socket.recv()
                image_np = np.frombuffer(image_bytes, dtype=np.uint8)
                image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

                if image_bgr is not None:
                    # Process the image and get AI predictions
                    throttle, steer = process_image_and_predict(image_bgr)
                    steer = -np.clip(steer, -1.0, 1.0) * 0.70
                    # Print steering and throttle values
                    print(f'Steering: {steer:.4f}, Throttle: {throttle:.4f}')
                    
                    # Create the command JSON and send it back to the client
                    commands = {"throttle": float(throttle), "steer": float(steer) }
                    socket.send_string(json.dumps(commands))
                else:
                    # Handle invalid image by sending a stop command
                    socket.send_string(json.dumps({"throttle": 0.0, "steer": 0.0}))

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                socket.send_string(json.dumps({"throttle": 0.0, "steer": 0.0}))

        # This keeps the OpenCV window responsive and checks for the quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    socket.close()
    context.term()
    cv2.destroyAllWindows()