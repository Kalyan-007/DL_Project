import cv2
import numpy
import os
from time import *
from pathlib import Path

# vc = cv2.VideoCapture(0)

# # Get the userId and userName
# print("Enter the sign : ")
# sign = input()
# sleep(5)

# # Initially Count is = 1
# count = 1


# # Function to save the image
# def saveImage(image, sign , imgId):
#     # Create a folder with the name as userName
    
#     print("[INFO] save called.")
#     Path("dataset/{}".format(sign)).mkdir(parents=True, exist_ok=True)
#     # Save the images inside the previously created folder
#     cv2.imwrite("dataset/{}/{}.jpg".format(sign, imgId), image)
#     print("[INFO] Image {} has been saved in folder : {}".format(imgId, sign))


# print("[INFO] Video Capture is now starting please stay still")

# while True:
#     key = cv2.waitKey(1) & 0xFF

#     sleep(2)
#     # Capture the frame/image
#     _, img = vc.read()
#     # Show the image
#     cv2.imshow("Identified Face", img)

#     # Wait for user keypress
    
#     # print("[INFO] key press ",key,str(ord('s')))

#     # Check if the pressed key is 'k' or 'q'
#     # if key == ord('s'):
#     # while count <50:
#         # If count is less than 5 then save the image
#     if count <= 30:
#         saveImage(img, sign, count)
#         count += 1
#     else:
#         break
#     # If q is pressed break out of the loop
#     # el
#     if key == ord('q'):
#         break

# print("[INFO] Dataset has been created for {}".format(sign))

# # Stop the video camera
# vc.release()
# # Close all Windows
# cv2.destroyAllWindows()
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
