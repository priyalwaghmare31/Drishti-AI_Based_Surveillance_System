from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from flask_cors import CORS
import os
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import pickle
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import time
import json
import psutil
import threading
import queue
import os
import traceback
from database import init_db, get_all_people, get_all_detections, get_person_by_name, increment_detection_count, add_person, load_face_db_from_db, signup_user, login_user, clear_all_persons, get_person_by_face_id, delete_person_by_face_id, get_db_connection
from face_utils import recognize_face_from_image

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app)  # Enable CORS for all routes

# Configuration
YOLO_MODEL = "yolov8n.pt"
DB_FILE = "face_db.pkl"
THRESHOLD = 0.4  # Increased threshold for better accuracy
RECOGNITION_HISTORY_FILE = "recognition_history.json"

# Global variables for real-time processing
detection_queue = queue.Queue()
processing_thread = None
is_processing = False

# Global live stats
live_stats = {
    'person_count': 0,
    'known_count': 0,
    'unknown_count': 0
}

# Performance optimization variables
frame_skip_counter = 0
FRAME_SKIP_RATE = 2  # Process every 2nd frame
face_recognition_cache = {}  # Cache for face recognition results
CACHE_TIMEOUT = 5  # Cache results for 5 seconds

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_face_db():
    try:
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        # Create empty database if not exists
        return {}

def save_face_db(face_db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(face_db, f)

def load_recognition_history():
    try:
        with open(RECOGNITION_HISTORY_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_recognition_history(history):
    with open(RECOGNITION_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def recognize_face(face_img, face_db):
    """Wrapper function to use the face_utils recognition function."""
    return recognize_face_from_image(face_img, face_db, threshold=THRESHOLD)



def recognize_face_from_image(face_img, face_db, threshold=0.4):
    """Recognize face from image using database embeddings"""
    try:
        # Generate embedding for the face image
        from face_utils import generate_embedding
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
            similarity = cosine_similarity(emb1, emb2)

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

def get_system_stats():
    """Get real system statistics"""
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        # GPU usage (if available)
        gpu_usage = 0
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
        except ImportError:
            pass

        disk_usage = psutil.disk_usage('/').percent
        network = psutil.net_io_counters()
        network_upload = network.bytes_sent // 1024  # KB/s
        network_download = network.bytes_recv // 1024  # KB/s

        return {
            'cpuUsage': cpu_usage,
            'memoryUsage': memory_usage,
            'gpuUsage': gpu_usage,
            'diskUsage': disk_usage,
            'networkUpload': network_upload,
            'networkDownload': network_download,
            'fps': 30,  # Target FPS
            'latency': 50  # Average latency in ms
        }
    except Exception as e:
        print(f"Error getting system stats: {e}")
        return {
            'cpuUsage': 0,
            'memoryUsage': 0,
            'gpuUsage': 0,
            'diskUsage': 0,
            'networkUpload': 0,
            'networkDownload': 0,
            'fps': 0,
            'latency': 0
        }

# Initialize SQLite database
init_db()

# Load models and database (with error handling for missing YOLO model)
try:
    model = YOLO(YOLO_MODEL)
    print("✅ YOLO model loaded successfully")
except Exception as e:
    print(f"⚠️ YOLO model loading failed: {e}")
    print("⚠️ Server will start without YOLO model - detection features will be limited")
    model = None

face_db = load_face_db_from_db()
print(f"Loaded face database with {len(face_db)} people")
recognition_history = load_recognition_history()

def gen_frames():
    """Enhanced video streaming with auto-reconnect and better error handling."""
    cap = None
    reconnect_attempts = 0
    max_reconnect_attempts = 5
    reconnect_delay = 2  # seconds

    def connect_camera():
        nonlocal cap, reconnect_attempts
        # Try different camera indices to find the correct one
        for i in range(10):  # Try cameras 0-9
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                print(f"✅ Camera opened successfully on index {i}")
                reconnect_attempts = 0
                return True
            cap = cv2.VideoCapture(i)  # Try without CAP_DSHOW
            if cap.isOpened():
                print(f"✅ Camera opened successfully on index {i} (without CAP_DSHOW)")
                reconnect_attempts = 0
                return True

        print("❌ Cannot open any camera - please check camera connections")
        return False

    # Initial camera connection
    if not connect_camera():
        print("❌ Failed to connect to camera initially")
        return

    frame_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 30  # About 1 second at 30fps

    while True:
        success, frame = cap.read()
        if not success:
            consecutive_failures += 1
            print(f"❌ Failed to read frame ({consecutive_failures}/{max_consecutive_failures})")

            if consecutive_failures >= max_consecutive_failures:
                print("🔄 Attempting to reconnect camera...")
                cap.release()
                time.sleep(reconnect_delay)

                if reconnect_attempts < max_reconnect_attempts:
                    reconnect_attempts += 1
                    if connect_camera():
                        consecutive_failures = 0
                        print("✅ Camera reconnected successfully")
                    else:
                        print(f"❌ Camera reconnection failed (attempt {reconnect_attempts}/{max_reconnect_attempts})")
                        if reconnect_attempts >= max_reconnect_attempts:
                            print("❌ Max reconnection attempts reached. Stopping video stream.")
                            break
                        time.sleep(reconnect_delay)
                        continue
                else:
                    print("❌ Max reconnection attempts reached. Stopping video stream.")
                    break
            time.sleep(0.1)
            continue

        consecutive_failures = 0  # Reset on success

        # Check if frame is valid
        if frame is None or frame.size == 0:
            print("⚠️ Empty frame received")
            time.sleep(0.1)
            continue

        # Optional: Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Perform detection and recognition
        if model is not None:
            try:
                results = model(frame, verbose=False)
                boxes = results[0].boxes

                person_count = 0
                known_count = 0
                unknown_count = 0

                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # person
                        person_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Extract face region with better bounds checking
                        face_crop = frame[max(0, y1-20):min(frame.shape[0], y2+20),
                                        max(0, x1-20):min(frame.shape[1], x2+20)]

                        if face_crop.size > 0 and face_crop.shape[0] > 10 and face_crop.shape[1] > 10:
                            name, face_confidence = recognize_face(face_crop, face_db)
                        else:
                            name, face_confidence = "Unknown", 0.0

                        if name != "Unknown":
                            known_count += 1
                            # Update database with detection
                            face_id = get_person_by_name(name)
                            if face_id:
                                increment_detection_count(face_id, metadata=f"Frame {frame_count}")
                        else:
                            unknown_count += 1

                        # Draw box + label
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        if name != "Unknown":
                            if face_confidence < 0.30:
                                label = f"{name} (Low Confidence)"
                            elif face_confidence > 1.0:
                                label = f"{name} (100%)"
                            else:
                                label = f"{name} ({int(face_confidence * 100)}%)"
                        else:
                            label = "Unknown"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Update global live stats
                live_stats['person_count'] = person_count
                live_stats['known_count'] = known_count
                live_stats['unknown_count'] = unknown_count

                # Show stats on frame
                cv2.putText(frame, f"Total: {person_count} | Known: {known_count} | Unknown: {unknown_count}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            except Exception as detection_error:
                print(f"⚠️ Detection error: {detection_error}")
                # Continue streaming even if detection fails
                cv2.putText(frame, "Detection Error", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("⚠️ Failed to encode frame")
            continue

        frame_bytes = buffer.tobytes()

        # Yield frame in multipart format for MJPEG streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        frame_count += 1

    if cap:
        cap.release()
    print("📹 Video stream ended")

@app.route('/video_feed')
def video_feed():
    return app.response_class(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect', methods=['POST'])
def detect():
    """Enhanced detection endpoint with consistent response format and live stats updates"""
    try:
        # Validate request data
        if not request.is_json:
            return jsonify({'success': False, 'message': 'Request must be JSON'}), 400

        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Missing image field in request'}), 400

        image_data = data['image']

        # Validate base64 data
        if not image_data or not isinstance(image_data, str):
            return jsonify({'success': False, 'message': 'Invalid image data format'}), 400

        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            if len(image_bytes) < 100:  # Minimum size check
                return jsonify({'success': False, 'message': 'Image too small for processing'}), 400

            image = Image.open(BytesIO(image_bytes))
            frame = np.array(image)

            # Validate image dimensions
            if frame.size == 0 or frame.shape[0] < 10 or frame.shape[1] < 10:
                return jsonify({'success': False, 'message': 'Image dimensions too small for detection'}), 400

        except Exception as decode_error:
            print(f"Image decode error: {decode_error}")
            return jsonify({'success': False, 'message': 'Invalid base64 image data'}), 400

        if model is None:
            return jsonify({'success': False, 'message': 'YOLO model not loaded'}), 500

        results = model(frame, verbose=False)
        boxes = results[0].boxes

        person_count = 0
        detections = []
        recognized_faces = []
        known_count = 0
        unknown_count = 0

        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Extract face region
                face_crop = frame[max(0, y1-20):y2+20, max(0, x1-20):x2+20]

                if face_crop.size > 0:
                    name, face_confidence = recognize_face(face_crop, face_db)
                else:
                    name, face_confidence = "Unknown", 0.0

                recognized_faces.append(name)

                # Update counts for live stats
                if name != "Unknown":
                    known_count += 1
                    # Update database with detection
                    face_id = get_person_by_name(name)
                    if face_id:
                        increment_detection_count(face_id, metadata="Manual detection")
                else:
                    unknown_count += 1

                # Create detection record
                detection = {
                    'id': len(detections),
                    'name': name,
                    'confidence': face_confidence,
                    'detection_confidence': confidence * 100,
                    'bbox': [x1, y1, x2, y2],
                    'timestamp': time.time()
                }
                detections.append(detection)

                # Add to recognition history
                if name != "Unknown" or face_confidence > 0:
                    history_entry = {
                        'id': len(recognition_history) + 1,
                        'name': name,
                        'confidence': face_confidence,
                        'timestamp': time.strftime('%H:%M:%S'),
                        'status': 'authorized' if name != 'Unknown' else 'unknown',
                        'action': 'detected'
                    }
                    recognition_history.append(history_entry)

                    # Keep only last 100 entries
                    if len(recognition_history) > 100:
                        recognition_history.pop(0)

        # Update global live stats
        live_stats['person_count'] = person_count
        live_stats['known_count'] = known_count
        live_stats['unknown_count'] = unknown_count

        # Save updated history
        save_recognition_history(recognition_history)

        return jsonify({
            'success': True,
            'message': 'Detection completed successfully',
            'data': {
                'person_count': person_count,
                'recognized_faces': recognized_faces,
                'detections': detections,
                'total_detections': len(detections)
            }
        })

    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({'success': False, 'message': f'Detection failed: {str(e)}'}), 500

@app.route('/live-stats', methods=['GET'])
def get_live_stats():
    """Get live detection statistics"""
    return jsonify(live_stats)

@app.route('/system-stats', methods=['GET'])
def get_system_stats_endpoint():
    """Get real-time system statistics"""
    stats = get_system_stats()
    return jsonify(stats)

@app.route('/recent-recognition', methods=['GET'])
def get_recent_recognition():
    """Get recent recognition activity - latest 10 entries"""
    try:
        detections = get_all_detections()
        # Sort by timestamp descending and take latest 10
        detections.sort(key=lambda x: x['last_seen'], reverse=True)
        recent_detections = detections[:10]

        # Convert to the expected format for frontend compatibility
        history = []
        for det in recent_detections:
            # Parse timestamp properly
            timestamp_str = det['last_seen']
            if ' ' in timestamp_str:
                # Format: "YYYY-MM-DD HH:MM:SS"
                date_part, time_part = timestamp_str.split(' ', 1)
                display_time = time_part
            else:
                display_time = timestamp_str

            history.append({
                'id': det['id'],
                'name': det['name'] or 'Unknown',
                'confidence': 0.0,  # Could be enhanced with actual confidence from detections table
                'timestamp': display_time,
                'status': 'authorized' if det['name'] and det['name'] != 'Unknown' else 'unknown',
                'action': 'detected'
            })

        # If no recent activity, return empty list (frontend will show "No recent activity")
        return jsonify(history)
    except Exception as e:
        print(f"Error fetching recent recognition: {e}")
        return jsonify([])

@app.route('/add-person', methods=['POST'])
def add_person_endpoint():
    """Add a new person with full data and image to the database"""
    try:
        # Get form data
        name = request.form.get('name')
        enroll = request.form.get('enroll')
        branch = request.form.get('branch')
        email = request.form.get('email')
        contact = request.form.get('contact')
        image_file = request.files.get('image')

        if not name:
            return jsonify({'success': False, 'message': 'Name is required'}), 400

        if not image_file:
            return jsonify({'success': False, 'message': 'Image file is required'}), 400

        # Call the database function
        result = add_person(
            name=name,
            enroll=enroll,
            branch=branch,
            email=email,
            contact=contact,
            image_file=image_file
        )

        if result['success']:
            # Reload face database after adding person
            global face_db
            face_db = load_face_db_from_db()
            return jsonify({'success': True, 'message': result['message'], 'data': {'face_id': result['face_id']}})
        else:
            return jsonify({'success': False, 'message': result['message']}), 400

    except Exception as e:
        print(f"Add person endpoint error: {e}")
        return jsonify({'success': False, 'message': f'Error adding person: {str(e)}'}), 500

@app.route('/api/add-person', methods=['POST'])
def api_add_person_endpoint():
    """Add a new person with full data and image to the database"""
    try:
        # Get form data
        name = request.form.get('name')
        enroll = request.form.get('enroll')  # studentId from frontend
        branch = request.form.get('branch')  # department from frontend
        email = request.form.get('email')
        contact = request.form.get('contact')  # phone from frontend
        image_file = request.files.get('image')

        if not name:
            return jsonify({'error': 'Name is required'}), 400

        if not image_file:
            return jsonify({'error': 'Image file is required'}), 400

        # Call the database function
        result = add_person(
            name=name,
            enroll=enroll,
            branch=branch,
            email=email,
            contact=contact,
            image_file=image_file
        )

        if result['success']:
            # Reload face database after adding person
            global face_db
            face_db = load_face_db_from_db()
            print(f"✅ Reloaded face database after adding {name}. Total people: {len(face_db)}")
            return jsonify({'success': True, 'message': result['message'], 'face_id': result['face_id']})
        else:
            # Return the specific error message from database function
            error_msg = result.get('error', 'Failed to add person to database')
            print(f"Add person failed: {error_msg}")
            return jsonify({'error': error_msg}), 400

    except Exception as e:
        print(f"Add person endpoint error: {e}")
        return jsonify({'error': f'Error adding person: {str(e)}'}), 500

@app.route('/api/persons', methods=['GET'])
def api_get_people():
    """API alias for people endpoint"""
    return get_people()

@app.route('/api/person/<int:face_id>', methods=['GET'])
def api_get_person(face_id):
    """Get person by face_id"""
    try:
        person = get_person_by_face_id(face_id)
        if person:
            return jsonify({
                'success': True,
                'message': 'Person found',
                'data': {
                    'face_id': person['face_id'],
                    'name': person['name'],
                    'enroll': person['enroll'],
                    'branch': person['branch'],
                    'email': person['email'],
                    'contact': person['contact'],
                    'image_path': person['image_path'],
                    'created_at': person['created_at']
                }
            })
        else:
            return jsonify({'success': False, 'message': 'Person not found'}), 404
    except Exception as e:
        print(f"Error fetching person {face_id}: {e}")
        return jsonify({'success': False, 'message': f'Error fetching person: {str(e)}'}), 500

@app.route('/api/clear-and-insert', methods=['POST'])
def api_clear_and_insert():
    """Clear all persons and detections from database"""
    try:
        result = clear_all_persons()
        if result['success']:
            # Reload face database after clearing
            global face_db
            face_db = load_face_db_from_db()
            return jsonify({'success': True, 'message': result['message']})
        else:
            return jsonify({'success': False, 'message': result['message']}), 500
    except Exception as e:
        print(f"Error clearing database: {e}")
        return jsonify({'success': False, 'message': f'Error clearing database: {str(e)}'}), 500

@app.route('/remove-person', methods=['POST'])
def remove_person():
    """Remove a person from the face database by face_id"""
    try:
        data = request.json
        face_id = data.get('face_id')

        if not face_id:
            return jsonify({'success': False, 'message': 'face_id is required'}), 400

        # Use the database function to delete by face_id
        result = delete_person_by_face_id(face_id)
        if result['success']:
            # Reload face database after removing person
            global face_db
            face_db = load_face_db_from_db()
            print(f"✅ Reloaded face database after removing person {face_id}. Total people: {len(face_db)}")
            return jsonify({'success': True, 'message': result['message']})
        else:
            return jsonify({'success': False, 'message': result['message']}), 404

    except Exception as e:
        print(f"Remove person error: {e}")
        return jsonify({'success': False, 'message': f'Error removing person: {str(e)}'}), 500

@app.route('/database-status', methods=['GET'])
def get_database_status():
    """Get database statistics from SQLite DB"""
    try:
        people = get_all_people()
        total_people = len(people)
        total_embeddings = sum(1 for p in people if p.get('embeddings'))

        people_list = []
        for person in people:
            people_list.append({
                'face_id': person['face_id'],
                'name': person['name'],
                'enroll': person['enroll'],
                'branch': person['branch'],
                'email': person['email'],
                'contact': person['contact'],
                'image_path': person.get('image_path'),
                'embeddings': person.get('embeddings') is not None,
                'created_at': person['created_at']
            })

        return jsonify({
            'total_people': total_people,
            'total_embeddings': total_embeddings,
            'people_list': people_list
        })
    except Exception as e:
        print(f"Error fetching database status: {e}")
        return jsonify({
            'total_people': 0,
            'total_embeddings': 0,
            'people_list': []
        })

@app.route('/people', methods=['GET'])
def get_people():
    """Get all people from the database"""
    try:
        people = get_all_people()
        people_list = []
        for person in people:
            people_list.append({
                'face_id': person['face_id'],
                'name': person['name'],
                'enroll': person['enroll'],
                'branch': person['branch'],
                'email': person['email'],
                'contact': person['contact'],
                'image_path': person.get('image_path'),
                'created_at': person['created_at']
            })
        return jsonify({'success': True, 'message': 'People retrieved successfully', 'data': people_list})
    except Exception as e:
        print(f"Error fetching people: {e}")
        return jsonify({'success': False, 'message': f'Error fetching people: {str(e)}'}), 500

@app.route('/api/signup', methods=['POST'])
def api_signup():
    """User signup endpoint with proper error messages"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')

        if not email or not password or not name:
            return jsonify({'error': 'Email, password, and name are required'}), 400

        result = signup_user(email, password, name)
        if result['success']:
            return jsonify({'success': True, 'message': result['message']})
        else:
            # Return specific error message and status code
            if 'Account already exists' in result['message']:
                return jsonify({'error': result['message']}), 409  # Conflict
            else:
                return jsonify({'error': result['message']}), 400
    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({'error': 'Signup failed'}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    """User login endpoint with proper error messages and session management"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        # Check if any users exist in the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users')
        user_count = cursor.fetchone()[0]

        if user_count == 0:
            return jsonify({'error': 'Account not found. Please create an account first.'}), 404

        result = login_user(email, password)
        if result['success']:
            # Set session for logged in user
            session['user_id'] = result['user']['id']
            session['user_email'] = result['user']['email']
            session['user_name'] = result['user']['name']
            return jsonify({'success': True, 'message': result['message']}), 200
        else:
            # Return specific error message and status code
            if 'No account found' in result['message']:
                return jsonify({'error': 'Email not registered. Please sign up first.'}), 404  # Not Found
            elif 'Incorrect password' in result['message']:
                return jsonify({'error': 'Incorrect password. Please try again.'}), 401  # Unauthorized
            else:
                return jsonify({'error': result['message']}), 401
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

# Keep old endpoints for backward compatibility
@app.route('/signup', methods=['POST'])
def signup():
    return api_signup()

@app.route('/login', methods=['POST'])
def login():
    return api_login()

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """User logout endpoint"""
    try:
        session.clear()
        return jsonify({'success': True, 'message': 'Logged out successfully'}), 200
    except Exception as e:
        print(f"Logout error: {e}")
        return jsonify({'error': 'Logout failed'}), 500

@app.route('/api/session', methods=['GET'])
def api_session():
    """Check current session status"""
    try:
        if 'user_id' in session:
            return jsonify({
                'logged_in': True,
                'user': {
                    'id': session['user_id'],
                    'email': session['user_email'],
                    'name': session['user_name']
                }
            }), 200
        else:
            return jsonify({'logged_in': False}), 200
    except Exception as e:
        print(f"Session check error: {e}")
        return jsonify({'error': 'Session check failed'}), 500

@app.route('/')
def dashboard():
    """Dashboard route with session check"""
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('dashboard.html')

@app.route('/login')
def login_page():
    """Login page route"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/signup')
def signup_page():
    """Signup page route"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/test-route')
def test_route():
    return "Test route is working!"

@app.route('/debug/embeddings', methods=['GET'])
def debug_embeddings():
    """Debug endpoint to check embeddings in database"""
    try:
        people = get_all_people()
        out = []
        for person in people:
            embeddings = person.get('embeddings')
            embeddings_parsed = None
            embeddings_type = None
            embeddings_len = 0

            if embeddings:
                try:
                    embeddings_parsed = json.loads(embeddings)
                    embeddings_type = type(embeddings_parsed).__name__
                    embeddings_len = len(embeddings_parsed) if isinstance(embeddings_parsed, list) else 0
                except json.JSONDecodeError:
                    embeddings_type = "invalid_json"
                    embeddings_len = len(embeddings)

            out.append({
                "face_id": person['face_id'],
                "name": person['name'],
                "email": person['email'],
                "embeddings_exists": embeddings is not None,
                "embeddings_type": embeddings_type,
                "embeddings_len": embeddings_len,
                "embeddings_preview": str(embeddings_parsed)[:100] if embeddings_parsed else None
            })
        return jsonify({"count": len(out), "people": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug/reload-db', methods=['POST'])
def debug_reload_db():
    """Debug endpoint to reload face database"""
    try:
        global face_db
        face_db = load_face_db_from_db()
        return jsonify({"message": f"Reloaded face database with {len(face_db)} people"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Drishti AI Surveillance System...")
    print(f"YOLOv8 Model: {YOLO_MODEL}")
    print(f"Face Database: {len(face_db)} people registered")
    print("Server running on http://localhost:5000")
    print("Backend available at http://localhost:5000")
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Please check if port 5000 is available or if there are missing dependencies.")
