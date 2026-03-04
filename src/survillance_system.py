# src/surveillance_system.py
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import pickle
import numpy as np
from database import increment_detection_count, get_person_by_name

# ------------------- SETTINGS -------------------
YOLO_MODEL = "yolov8n.pt"
DB_FILE = "face_db.pkl"
THRESHOLD = 0.4  # Face similarity threshold

# ------------------- HELPERS -------------------
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_face_db():
    with open(DB_FILE, "rb") as f:
        return pickle.load(f)

def recognize_face(face_img, face_db):
    """Return the name of the recognized person or 'Unknown'"""
    try:
        reps = DeepFace.represent(face_img, model_name="VGG-Face", enforce_detection=False)
        if not reps:
            return "Unknown"

        embedding = reps[0]["embedding"]
        best_match, best_score = "Unknown", -1

        for person, db_embeddings in face_db.items():
            for db_emb in db_embeddings:
                score = cosine_similarity(embedding, db_emb)
                if score > best_score:
                    best_score, best_match = score, person

        return best_match if best_score >= THRESHOLD else "Unknown"
    except:
        return "Unknown"

# ------------------- MAIN -------------------
def main():
    # Load YOLO + face DB
    model = YOLO(YOLO_MODEL)
    face_db = load_face_db()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("Press 'q' to quit")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        boxes = results[0].boxes

        person_count = 0

        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Crop face region
                face_crop = frame[y1:y2, x1:x2]
                name = recognize_face(face_crop, face_db) if face_crop.size > 0 else "Unknown"

                # Update database with detection
                if name != "Unknown":
                    face_id = get_person_by_name(name)
                    if face_id:
                        increment_detection_count(face_id, metadata=f"Frame {frame_count}")

                # Draw box + label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show person count
        cv2.putText(frame, f"Person Count: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Smart Surveillance System", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Surveillance ended")

if __name__ == "__main__":
    main()
