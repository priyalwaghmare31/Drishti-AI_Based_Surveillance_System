# src/build_face_db.py
import os
from deepface import DeepFace
import pickle

DATASET_PATH = "data/faces"
DB_FILE = "face_db.pkl"

def build_database():
    face_db = {}

    print("📸 Building face embeddings database...")

    for person_name in os.listdir(DATASET_PATH):
        person_dir = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)

            try:
                # Extract embedding (vector representation of the face)
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name="VGG-Face",   # you can change to VGG-Face, ArcFace, etc.
                    enforce_detection=False
                )
                if embedding and isinstance(embedding, list):
                    embeddings.append(embedding[0]["embedding"])
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")

        if embeddings:
            face_db[person_name] = embeddings
            print(f"✅ Added {person_name} with {len(embeddings)} images")

    # Save database to pickle file
    with open(DB_FILE, "wb") as f:
        pickle.dump(face_db, f)

    print(f"🎉 Database built and saved as {DB_FILE}")

if __name__ == "__main__":
    build_database()
