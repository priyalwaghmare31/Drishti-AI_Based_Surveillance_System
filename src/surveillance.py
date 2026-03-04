import cv2
import numpy as np
from face_utils import match_face

def process_frame(frame, model, known_embeddings, threshold):
    """Process frame with YOLO detection and face recognition"""
    if model is None:
        return frame

    try:
        # Run YOLO detection
        results = model(frame, verbose=False)
        boxes = results[0].boxes

        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Extract face region
                face_crop = frame[max(0, y1):min(frame.shape[0], y2),
                                max(0, x1):min(frame.shape[1], x2)]

                if face_crop.size > 0 and face_crop.shape[0] > 50 and face_crop.shape[1] > 50:
                    # Generate embedding for detected face
                    from face_utils import generate_embedding
                    embedding = generate_embedding(face_crop)

                    if embedding is not None:
                        # Match against known faces
                        match = match_face(embedding, known_embeddings, threshold)

                        if match:
                            name = match['name']
                            color = (0, 255, 0)  # Green for known
                            label = f"{name}"
                            # Record detection
                            from database import get_person_by_name, increment_detection_count
                            face_id = get_person_by_name(name)
                            if face_id:
                                increment_detection_count(face_id)
                        else:
                            name = "Unknown"
                            color = (0, 0, 255)  # Red for unknown
                            label = "Unknown"
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)
                        label = "Unknown"
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    label = "Unknown"

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame
