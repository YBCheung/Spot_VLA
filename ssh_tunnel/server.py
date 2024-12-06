import requests
import numpy as np
import json
import time 
import cv2

# Function to send data to the server
def send_data_to_server(image_array, prompt, array_data):
    # array, string, array

    # Encode the image to JPEG format (you can choose PNG or other formats)
    success, encoded_image = cv2.imencode('.jpg', image_array)

    if not success:
        print('invalid image, need array')
        return None, None
    
    # Convert the encoded image to binary data
    image_binary = encoded_image.tobytes()
    # Send POST request to the server
    url = 'http://localhost:8888/process'  # Replace with your server IP and port
    files = {'image': ('image.jpg', image_binary, 'image/jpeg')} # need json?
    data = {'state': json.dumps(array_data.tolist()), 'prompt': prompt}  # Convert array to list and then JSON format

    # Make the request
    response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        return result['action'], result['feedback']
    else:
        print("Failed to get response from server:", response.status_code)
        return None, None

# Example usage
image_array = np.random.rand(224, 224, 3)  # Replace with actual image array
prompt = 'move the blue bottle in the box'
array_data = np.array([0, 1, 2, 3, 4, 5, 6])  # Example array
for i in range(10):
    start_time = time.time()
    processed_array, output_string = send_data_to_server(image_array, prompt, array_data)
    print(f'FREQ={1 / (time.time() - start_time)}')
    print(processed_array, output_string)
