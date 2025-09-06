import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
print("Pong server started. Waiting for a ping...")

while True:
    message = socket.recv_string()
    print(f"Received: '{message}'")
    socket.send_string("pong")
    print("Sent: 'pong'")