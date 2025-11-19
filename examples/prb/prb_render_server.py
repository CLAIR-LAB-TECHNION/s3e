import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))

import socket
import pickle

from photorealistic_blocksworld.render_utils import render_scene

import subprocess

def is_gpu_available():
    try:
        # Try to run the nvidia-smi command
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # If the return code is 0, the GPU is available
        return result.returncode == 0
    except FileNotFoundError:
        # nvidia-smi not found, likely no NVIDIA GPU
        return False

DEFAULT_PORT = 65432

def render_server(port: int = DEFAULT_PORT):
    # check GPU availability
    print('checking GPU availability')
    gpu_available = is_gpu_available()
    print('GPU available:', gpu_available)
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', port))  # Ensure this matches the controller's settings
        s.listen()

        print("Renderer is listening for requests on port", port)
        sys.stdout.flush()

        # Wait for a connection
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
                
                # add gpu usage parameter
                # GPU OPTION DOESN"T WORK
                # request['args'].use_gpu = int(gpu_available)

                try:
                    # Process the render request
                    render_scene(**request)

                    # Send back confirmation
                    conn.sendall(b"done")
                except Exception as e:
                    # send back failure message
                    conn.sendall(str(e))

if __name__ == "__main__":
    port = int(sys.argv[-1])
    render_server(port)
