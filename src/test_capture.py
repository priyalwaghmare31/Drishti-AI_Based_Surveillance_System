import cv2

def main():
    src = 0  # use 0 for webcam, or replace with "data/samples/sample.mp4"
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print("ERROR: Cannot open video source", src)
        return

    print("Press 'q' to exit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or can't read frame.")
            break

        cv2.putText(frame, "Test capture - press q to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
