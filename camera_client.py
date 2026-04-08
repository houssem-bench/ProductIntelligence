import cv2
import requests
from datetime import datetime
import os

# 🔴 Replace with your real ngrok HTTPS URL
NGROK_URL = "https://subattenuate-joanne-vacuous.ngrok-free.dev"
endpoint = f"{NGROK_URL}/analyze"

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press SPACE to take picture, ESC to exit.")

# Create folder to store images
os.makedirs("photos", exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    # SPACE key to capture
    if key == 32:
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create unique filename
        image_path = f"photos/captured_{timestamp}.jpg"

        cv2.imwrite(image_path, frame)
        print(f"Image saved as {image_path}")

        # Upload to server
        with open(image_path, "rb") as img:
            files = {"image": img}
            response = requests.post(endpoint, files=files)

        print("Status:", response.status_code)
        print("Response:")
        print(response.text)

        break

    # ESC key to exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()