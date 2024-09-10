import cv2
import requests
import numpy as np

API_URL = "http://127.0.0.1:8000/" 

CAMERA_INDEX = 0 
KEY_CAPTURE = ord('s')  # Key to press to capture the image ('s')

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

def send_image_to_api(image):
    """Send the image to the FastAPI endpoint."""
    _, img_encoded = cv2.imencode('.jpg', image)

    img_bytes = img_encoded.tobytes()
    response = requests.post(API_URL+'spoof', files={"file": ("image.jpg", img_bytes, "image/jpeg")})
    not_spoof = response.json()['Spoof']
    print('Spoof:', not_spoof)

    if not_spoof:
        response = requests.post(API_URL+'predict', files={"file": ("image.jpg", img_bytes, "image/jpeg")})
        print(response.json())
    
    
def main():
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        # Display the frame
        cv2.imshow('Webcam', frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == KEY_CAPTURE:
            print("Capture key pressed. Taking picture...")
            send_image_to_api(frame)
        
        # Exit if 'q' key is pressed
        if key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
