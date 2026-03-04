import cv2
import time

# Test camera access
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

print("Camera opened successfully")
print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

ret, frame = cap.read()
if ret:
    print("Frame captured successfully")
    cv2.imwrite("test_frame.jpg", frame)
    print("Saved test_frame.jpg")
else:
    print("Failed to capture frame")

cap.release()
