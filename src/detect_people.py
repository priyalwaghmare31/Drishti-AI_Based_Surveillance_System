from ultralytics import YOLO
import cv2

def main():
    # Load YOLO model
    model = YOLO("yolov8n.pt")   # COCO pretrained

    # Try opening webcam (use CAP_DSHOW for Windows, fallback to default otherwise)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # fallback
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read frame")
            break

        # Run YOLO detection
        results = model(frame, verbose=False)
        boxes = results[0].boxes
        person_count = 0

        for box in boxes:
            cls = int(box.cls[0])  # class id
            if cls == 0:  # person in COCO dataset
                person_count += 1
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show count at top
        cv2.putText(frame, f"Person Count: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Show window
        cv2.imshow("Person Detection & Count", frame)

        # Wait for key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Capture ended")

if __name__ == "__main__":
    main()
