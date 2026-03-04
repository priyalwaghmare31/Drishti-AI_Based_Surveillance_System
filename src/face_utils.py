import numpy as np
import cv2
from deepface import DeepFace
import json
from database import get_all_people

def generate_embedding(image):
    """Generate face embedding using DeepFace"""
    try:
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Generate embedding using DeepFace
        embedding = DeepFace.represent(
            img_path=image,
            model_name='Facenet',
            enforce_detection=True,
            detector_backend='mtcnn'
        )

        if isinstance(embedding, list) and len(embedding) > 0:
            return np.array(embedding[0]['embedding'])
        else:
            return None
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def load_known_embeddings():
    """Load all known embeddings from database"""
    try:
        people = get_all_people()
        embeddings = {}

        for person in people:
            if person['embedding']:
                try:
                    # Parse JSON string back to numpy array
                    emb_list = json.loads(person['embedding'])
                    embeddings[person['face_id']] = {
                        'embedding': np.array(emb_list),
                        'name': person['name'],
                        'enroll': person['enroll']
                    }
                except json.JSONDecodeError as e:
                    print(f"Error parsing embedding for {person['name']}: {e}")
                    continue

        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return {}

def match_face(embedding, known_embeddings, threshold=0.6):
    """Match face embedding against known embeddings using L2 distance"""
    if embedding is None or len(known_embeddings) == 0:
        return None

    min_distance = float('inf')
    best_match = None

    for face_id, data in known_embeddings.items():
        known_emb = data['embedding']

        # Calculate L2 distance
        distance = np.linalg.norm(embedding - known_emb)

        if distance < min_distance:
            min_distance = distance
            best_match = data

    # Check if match is within threshold
    if min_distance < threshold:
        # Calculate confidence as percentage (inverse of normalized distance)
        confidence = max(0, min(1, 1 - (min_distance / threshold))) * 100
        return {**best_match, 'confidence': confidence}
    else:
        return None

def recognize_face_from_image(face_img, face_db, threshold=0.4):
    """Recognize face from image using database embeddings"""
    try:
        # Generate embedding for the face image
        embedding = generate_embedding(face_img)

        if embedding is None:
            return "Unknown", 0.0

        # Find best match in face database
        best_match = None
        best_similarity = 0.0

        for name, known_embedding in face_db.items():
            # Convert to numpy arrays
            emb1 = np.array(embedding)
            emb2 = np.array(known_embedding)

            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name

        # Check if match is above threshold
        if best_similarity >= threshold:
            return best_match, best_similarity
        else:
            return "Unknown", best_similarity

    except Exception as e:
        print(f"Error in face recognition: {e}")
        return "Unknown", 0.0
