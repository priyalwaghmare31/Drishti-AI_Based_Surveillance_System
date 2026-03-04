# src/face_recognition_deepface_db.py
import cv2
from deepface import DeepFace
import pickle
import numpy as np

DB_FILE = "face_db.pkl"
THRESHOLD = 0.4  # similarity threshold (lower = stricter, adjust as needed)

# Load the database
with open(DB_FILE, "rb") as f:
    face_db = pickle.load(f)

print(f"✅ Loaded database with {len(face_db)} people")


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def recognize_face(frame):
    try:
        # Get embeddings for faces in the frame
        reps = DeepFace.represent(
            frame,
            model_name="VGG-Face",   # should match the model used in build_face_db
            enforce_detection=False
        )

        names = []
        for rep in reps:
            embedding = rep["embedding"]

            best_match = "Unknown"
            best_score = -1

            # Compare against database
            for person, db_embeddings in face_db.items():
                for db_emb in db_embeddings:
                    score = cosine_similarity(embedding, db_emb)
                    if score > best_score:
                        best_score = score
                        best_match = person

            # Apply threshold
            if best_score < THRESHOLD:
                best_match = "Unknown"

            names.append((best_match, best_score))

        return names

    except Exception as e:
        print("⚠️ DeepFace error:", e)
        return []


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = recognize_face(frame)

        # Show results
        y = 40
        for name, score in results:
            cv2.putText(frame, f"{name} ({score:.2f})", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 40

        cv2.imshow("Face Recognition (DeepFace + DB)", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Ended")


if __name__ == "__main__":
    main()
