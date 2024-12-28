import cv2
import numpy
import os
from time import *
from pathlib import Path

def main():
    # Open the webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press SPACE to capture a photo. Press 'q' to quit.")
    photo_count = 1

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Spacebar to capture photo
            photo_filename = f"1_{photo_count}.jpg"
            cv2.imwrite(photo_filename, frame)
            print(f"Photo saved as {photo_filename}")
            photo_count += 1

        elif key == ord('q'):  # 'q' to quit
            print("Exiting...")
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
