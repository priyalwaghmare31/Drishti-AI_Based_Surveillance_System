from flask import Flask, request, jsonify, render_template, Response, session, redirect, url_for
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import time
import json
import os
from database import init_db, get_all_people, add_person, delete_person_by_face_id, signup_user, login_user
from face_utils import generate_embedding, load_known_embeddings, match_face
from surveillance import process_frame

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
CORS(app)

# Configuration
YOLO_MODEL = "yolov8n.pt"
THRESHOLD = 0.6

# Global variables
model = None
known_embeddings = {}

def init_app():
    global model, known_embeddings
    try:
        model = YOLO(YOLO_MODEL)
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"⚠️ YOLO model loading failed: {e}")
        model = None

    # Initialize database
    init_db()

    # Load known embeddings
    known_embeddings = load_known_embeddings()
    print(f"Loaded {len(known_embeddings)} known embeddings")

init_app()

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('dashboard.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/signup')
def signup_page():
    return render_template('signup.html')

@app.route('/add_person', methods=['POST'])
def add_person_endpoint():
    """Add a new person with image"""
    try:
        name = request.form.get('name')
        enroll = request.form.get('enroll', '')
        branch = request.form.get('branch', '')
        email = request.form.get('email', '')
        contact = request.form.get('contact', '')
        image_file = request.files.get('image')

        if not name or not image_file:
            return jsonify({'success': False, 'message': 'Name and image are required'}), 400

        # Generate embedding
        image = Image.open(image_file)
        embedding = generate_embedding(np.array(image))

        if embedding is None:
            return jsonify({'success': False, 'message': 'Failed to generate face embedding'}), 400

        # Save to database
        result = add_person(
            name=name,
            enroll=enroll,
            branch=branch,
            email=email,
            contact=contact,
            embedding=json.dumps(embedding.tolist()),
            image_path=''  # Could save image file if needed
        )

        if result['success']:
            # Reload embeddings immediately
            global known_embeddings
            known_embeddings = load_known_embeddings()
            return jsonify({'success': True, 'message': result['message'], 'face_id': result['face_id']})
        else:
            return jsonify({'success': False, 'message': result['message']}), 400

    except Exception as e:
        print(f"Add person error: {e}")
        return jsonify({'success': False, 'message': f'Error adding person: {str(e)}'}), 500

@app.route('/get_persons', methods=['GET'])
def get_persons():
    """Get all persons"""
    try:
        people = get_all_people()
        return jsonify({'success': True, 'data': people})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """MJPEG video stream"""
    def generate():
        cap = cv2.VideoCapture(0)
        # Optimize camera settings for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            return

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Process frame with YOLO and face recognition
            processed_frame = process_frame(frame, model, known_embeddings, THRESHOLD)

            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'}), 400

        result = login_user(email, password)
        if result['success']:
            session['user_id'] = result['user']['id']
            session['email'] = result['user']['email']
            return jsonify({'success': True, 'message': result['message']})
        else:
            return jsonify({'success': False, 'message': result['message']}), 401
    except Exception as e:
        return jsonify({'success': False, 'message': 'Login failed'}), 500

@app.route('/signup', methods=['POST'])
def signup():
    """User signup"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')

        if not email or not password or not name:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400

        result = signup_user(email, password, name)
        if result['success']:
            return jsonify({'success': True, 'message': result['message']})
        else:
            return jsonify({'success': False, 'message': result['message']}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': 'Signup failed'}), 500

@app.route('/get_recent_detections', methods=['GET'])
def get_recent_detections():
    """Get recent detections with names"""
    try:
        from database import get_recent_detections_with_names
        detections = get_recent_detections_with_names()
        return jsonify({'success': True, 'data': detections})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/debug/embeddings', methods=['GET'])
def debug_embeddings():
    """Debug endpoint to check embeddings"""
    try:
        people = get_all_people()
        return jsonify({'count': len(people), 'embeddings': known_embeddings})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
