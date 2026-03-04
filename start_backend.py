#!/usr/bin/env python3
"""
Drishti AI Surveillance System - Backend Startup Script
"""

import os
import sys
import subprocess
import time

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import flask
        import ultralytics
        import deepface
        import cv2
        import numpy
        import PIL
        print("✓ All Python dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def download_models():
    """Download required AI models"""
    try:
        from ultralytics import YOLO
        print("Downloading YOLOv8 model...")
        model = YOLO("yolov8n.pt")
        print("✓ YOLOv8 model downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error downloading models: {e}")
        return False

def start_server():
    """Start the Flask server"""
    print("Starting Drishti AI Surveillance System...")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    try:
        os.chdir("src")
        subprocess.run([sys.executable, "server.py"], check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Server error: {e}")

def main():
    print("Drishti AI Surveillance System")
    print("=" * 40)

    # Check requirements
    if not check_requirements():
        return

    # Download models
    if not download_models():
        return

    # Initialize database
    try:
        from src.database import init_db, seed_demo_people
        init_db()
        demo_people = [
            {'name': 'Priyal Waghmare', 'enroll': '0827CS221230', 'branch': 'CSE', 'email': 'priyal@example.com', 'contact': '1234567890'},
            {'name': 'Reeti Shrimal', 'enroll': '0827CS221223', 'branch': 'CSE', 'email': 'reeti@example.com', 'contact': '1234567891'},
            {'name': 'Vidhi Shrimal', 'enroll': '0827CS221222', 'branch': 'CSE', 'email': 'vidhi@example.com', 'contact': '1234567892'}
        ]
        seed_demo_people(demo_people)
        print("✓ Database initialized and seeded with demo data")
    except Exception as e:
        print(f"✗ Database initialization error: {e}")
        return

    # Start server
    start_server()

if __name__ == "__main__":
    main()
