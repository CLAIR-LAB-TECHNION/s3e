import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))

import socket
import pickle

from photorealistic_blocksworld.render_utils import render_scene


def render_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 65432))  # Ensure this matches the controller's settings
        s.listen()
        print("Renderer is listening for requests...")
        while True:
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                data = conn.recv(4096)
                print('received data')
                if not data:
                    print('breaking???')
                    break
                print('unpickling data')
                request = pickle.loads(data)
                print('unpickled data')
                
                try:
                    # Process the render request
                    render_scene(**request)

                    # Send back confirmation
                    conn.sendall(b"done")
                except Exception as e:
                    # send back failure message
                    conn.sendall(str(e))

if __name__ == "__main__":
    render_server()
