## Setting Up the SSH Tunnel Connection

1. check remote machine available ports
       
    ```powershell
    $ netstat -tuln | grep LISTEN

    tcp        0      0 127.0.0.1:8072          0.0.0.0:*               LISTEN     
    tcp        0      0 127.0.0.1:8099          0.0.0.0:*               LISTEN     
    tcp        0      0 127.0.0.1:25            0.0.0.0:*               LISTEN     
    tcp        0      0 127.0.0.1:9001          0.0.0.0:*               LISTEN     
    tcp6       0      0 :::33939                :::*                    LISTEN     
    tcp6       0      0 :::8086                 :::*                    LISTEN     
    tcp6       0      0 :::111                  :::*                    LISTEN     
    tcp6       0      0 :::9100                 :::*                    LISTEN 
    ```
    The command shows a list of ports that are **already in use** for listening, meaning **you should not use** these ports for your new tunnel. 
    
    ```powershell
    $ nc -zv localhost <port_number>
    ```
    This is to test if the desired port is available. If it is in use, it will connect successfully. 
    
    ```powershell
    # 9001 is occupied
    $ nc -zv localhost 9001
    nc: connect to localhost (::1) port 9001 (tcp) failed: Connection refused
    Connection to localhost (127.0.0.1) 9001 port [tcp/etlservicemgr] succeeded!
    
    # 9024 is free
    $ nc -zv localhost 9024
    nc: connect to localhost (::1) port 9024 (tcp) failed: Connection refused
    nc: connect to localhost (127.0.0.1) port 9024 (tcp) failed: Connection refused
    ```
    
    
    
    Or run the following, if no output, then <port_number> is not occupied.
    
    ```powershell
    netstat -tuln | grep <port_number>
    ```
    
2. Check free local ports
    
    The same rule as mentioned above. 
    
3. Establish the ssh tunnel, keep it running. 

    ```powershell
    ssh -L 8888:localhost:9024 <user>@<triton_server>
    ```

4. run code on the server, listening consistently to the local PC port. stay idle if no request. Remember to change the server port (9024 in the example code)
    
    ```python
    # compress image to jpeg to save bandwidth. 100-160Hz achieved. 
    from flask import Flask, request, jsonify
    import numpy as np
    import json
    import cv2

    app = Flask(__name__)

    @app.route('/process', methods=['POST'])
    def process_image():
        image_file = request.files['image']      # FileStorage object
        if image_file:
            # Convert bytes to a NumPy array
            image_array = np.frombuffer(image_file.read(), np.uint8)

            # Decode the image
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            print(image.shape)

        # Retrieve the array data
        array_data_json = request.form['state']
        array_data = np.array(json.loads(array_data_json))  # Convert JSON back to array


        # Process it using your VLM model
        processed_result, result_string = [6,5,4,3,2,1,0], "Tunnel test success!"
        
        # Return processed result and a string
        return jsonify({'action': processed_result, 'VLM_string': result_string})

    if __name__ == "__main__":
        app.run(host='0.0.0.0', port=9024)  # Run the server on port 9024

    ```
    
    then run “netstat”, it should show your port is listening. 
    
    ```powershell
    $ netstat -tuln | grep 9024
    
    tcp        0      0 0.0.0.0:9024            0.0.0.0:*               LISTEN  
    ```
    
2. run code on the local PC. Remember to modify the local machine port (8888 in the example code)
    
    ```python
    # compress image to jpeg to save bandwidth. FREQ 100-160Hz achieved. 
    
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
        files = {'image': ('image.jpg', image_binary, 'image/jpeg')}
        data = {'state': json.dumps(array_data.tolist()), 'prompt': prompt}  # Convert array to list and then JSON format

        # Make the request
        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            return result['action'], result['VLM_string']
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


    ```
    

then the local machine terminal should display:

```powershell
Received: {'VLM_string': 'Tunnel test success!', 'action': [6, 5, 4, 3, 2, 1, 0]}
FREQ=123.78785821798542
Received: {'VLM_string': 'Tunnel test success!', 'action': [6, 5, 4, 3, 2, 1, 0]}
FREQ=117.05469970975665
Received: {'VLM_string': 'Tunnel test success!', 'action': [6, 5, 4, 3, 2, 1, 0]}
FREQ=110.39675729739689
Received: {'VLM_string': 'Tunnel test success!', 'action': [6, 5, 4, 3, 2, 1, 0]}
FREQ=132.77315606204496
```

on the server terminal should display:

```powershell
127.0.0.1 - - [17/Oct/2024 21:52:28] "POST /process HTTP/1.1" 200 -
(224, 224, 3)
127.0.0.1 - - [17/Oct/2024 21:52:28] "POST /process HTTP/1.1" 200 -
(224, 224, 3)
127.0.0.1 - - [17/Oct/2024 21:52:29] "POST /process HTTP/1.1" 200 -
(224, 224, 3)
127.0.0.1 - - [17/Oct/2024 21:52:29] "POST /process HTTP/1.1" 200 -
```

### Possible problems:

1. Connection refused:
    
    ```powershell
    Error: HTTPConnectionPool(host='localhost', port=8888): Max retries exceeded with url: /process (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f82be2d2170>: Failed to establish a new connection: [Errno 111] Connection refused'))
    ```
    
    The tunnel connection failed. After running “ssh -L …”, the session should not be closed. Run Flake on the server. then run local code.