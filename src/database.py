import sqlite3
import json
import os
import time

DB_PATH = os.path.join(os.path.dirname(__file__), 'surveillance.db')

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create persons table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            face_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            enroll TEXT,
            branch TEXT,
            email TEXT,
            contact TEXT,
            embedding TEXT,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create detections table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (face_id) REFERENCES persons (face_id)
        )
    ''')

    conn.commit()
    conn.close()

def add_person(name, enroll, branch, email, contact, embedding=None, image_path=None, image_file=None):
    """Add a new person to database"""
    try:
        # Handle image file if provided
        if image_file:
            # Save image file
            import os
            from werkzeug.utils import secure_filename

            # Create images directory if it doesn't exist
            images_dir = 'data/images'
            os.makedirs(images_dir, exist_ok=True)

            # Generate unique filename
            filename = secure_filename(f"{name.replace(' ', '_')}_{int(time.time())}.jpg")
            image_path = os.path.join(images_dir, filename)

            # Save the image
            image_file.save(image_path)

            # Generate embedding if not provided
            if embedding is None:
                from face_utils import generate_embedding
                import cv2
                image = cv2.imread(image_path)
                if image is not None:
                    embedding = generate_embedding(image)
                    if embedding is not None:
                        embedding = json.dumps(embedding.tolist())

        # Convert embedding to JSON string if it's a numpy array
        if embedding is not None and not isinstance(embedding, str):
            if hasattr(embedding, 'tolist'):
                embedding = json.dumps(embedding.tolist())
            else:
                embedding = json.dumps(embedding)

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO persons (name, enroll, branch, email, contact, embedding, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, enroll, branch, email, contact, embedding, image_path))

        face_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return {'success': True, 'message': f'Person {name} added successfully', 'face_id': face_id}
    except Exception as e:
        return {'success': False, 'message': str(e)}

def get_all_people():
    """Get all persons from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM persons')
        people = cursor.fetchall()

        result = []
        for person in people:
            result.append({
                'face_id': person['face_id'],
                'name': person['name'],
                'enroll': person['enroll'],
                'branch': person['branch'],
                'email': person['email'],
                'contact': person['contact'],
                'embedding': person['embedding'],
                'image_path': person['image_path'],
                'created_at': person['created_at']
            })

        conn.close()
        return result
    except Exception as e:
        print(f"Error getting people: {e}")
        return []

def delete_person_by_face_id(face_id):
    """Delete person by face_id"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM persons WHERE face_id = ?', (face_id,))
        conn.commit()
        conn.close()

        return {'success': True, 'message': f'Person {face_id} deleted successfully'}
    except Exception as e:
        return {'success': False, 'message': str(e)}

def signup_user(email, password, name):
    """Create new user account"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            return {'success': False, 'message': 'Account already exists'}

        # Create user
        cursor.execute('''
            INSERT INTO users (email, password, name)
            VALUES (?, ?, ?)
        ''', (email, password, name))

        conn.commit()
        conn.close()

        return {'success': True, 'message': 'Account created successfully'}
    except Exception as e:
        return {'success': False, 'message': str(e)}

def login_user(email, password):
    """Authenticate user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # First check if email exists
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()

        if not user:
            conn.close()
            return {'success': False, 'message': 'Account does not exist'}

        # Check password
        cursor.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password))
        user_with_password = cursor.fetchone()

        conn.close()

        if user_with_password:
            return {'success': True, 'message': 'Login successful', 'user': dict(user_with_password)}
        else:
            return {'success': False, 'message': 'Incorrect Password'}
    except Exception as e:
        return {'success': False, 'message': str(e)}

def get_person_by_name(name):
    """Get person by name"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM persons WHERE name = ?', (name,))
        person = cursor.fetchone()

        conn.close()

        if person:
            return person['face_id']
        else:
            return None
    except Exception as e:
        print(f"Error getting person by name: {e}")
        return None

def increment_detection_count(face_id, metadata=None):
    """Increment detection count for a person"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('INSERT INTO detections (face_id) VALUES (?)', (face_id,))

        conn.commit()
        conn.close()

        return {'success': True, 'message': 'Detection count incremented'}
    except Exception as e:
        return {'success': False, 'message': str(e)}

def get_recent_detections_with_names(limit=10):
    """Get recent detections with person names"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT d.id, d.face_id, d.timestamp, p.name
            FROM detections d
            JOIN persons p ON d.face_id = p.face_id
            ORDER BY d.timestamp DESC
            LIMIT ?
        ''', (limit,))

        detections = cursor.fetchall()

        result = []
        for det in detections:
            result.append({
                'id': det['id'],
                'face_id': det['face_id'],
                'name': det['name'],
                'timestamp': det['timestamp']
            })

        conn.close()
        return result
    except Exception as e:
        print(f"Error getting recent detections: {e}")
        return []

def get_all_detections():
    """Get all detections with person names and last seen timestamps"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT d.id, d.face_id, d.timestamp as last_seen, p.name
            FROM detections d
            JOIN persons p ON d.face_id = p.face_id
            ORDER BY d.timestamp DESC
        ''')

        detections = cursor.fetchall()

        result = []
        for det in detections:
            result.append({
                'id': det['id'],
                'face_id': det['face_id'],
                'name': det['name'],
                'last_seen': det['last_seen']
            })

        conn.close()
        return result
    except Exception as e:
        print(f"Error getting all detections: {e}")
        return []

def clear_all_persons():
    """Clear all persons and detections"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM detections')
        cursor.execute('DELETE FROM persons')

        conn.commit()
        conn.close()

        return {'success': True, 'message': 'All persons and detections cleared'}
    except Exception as e:
        return {'success': False, 'message': str(e)}

def get_person_by_face_id(face_id):
    """Get person by face_id"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM persons WHERE face_id = ?', (face_id,))
        person = cursor.fetchone()

        conn.close()

        if person:
            return {
                'face_id': person['face_id'],
                'name': person['name'],
                'enroll': person['enroll'],
                'branch': person['branch'],
                'email': person['email'],
                'contact': person['contact'],
                'embedding': person['embedding'],
                'image_path': person['image_path'],
                'created_at': person['created_at']
            }
        else:
            return None
    except Exception as e:
        print(f"Error getting person by face_id: {e}")
        return None

def load_face_db_from_db():
    """Load face database from SQLite DB with improved error handling"""
    try:
        people = get_all_people()
        face_db = {}

        for person in people:
            if person.get('embedding'):
                try:
                    # Parse JSON string to numpy array
                    embedding_list = json.loads(person['embedding'])
                    if isinstance(embedding_list, list) and len(embedding_list) > 0:
                        # Convert to numpy array for consistency
                        import numpy as np
                        face_db[person['name']] = np.array(embedding_list)
                    else:
                        print(f"Invalid embedding format for {person['name']}: not a valid list")
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for {person['name']}: {e}")
                except Exception as e:
                    print(f"Error processing embedding for {person['name']}: {e}")

        print(f"Successfully loaded {len(face_db)} face embeddings from database")
        return face_db
    except Exception as e:
        print(f"Error loading face DB: {e}")
        return {}
