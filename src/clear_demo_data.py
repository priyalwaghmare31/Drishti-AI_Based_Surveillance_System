#!/usr/bin/env python3
"""
Script to clear all demo data from the surveillance system database.
This removes all existing people, detections, and associated files.
"""

from database import clear_all_persons, init_db
import os

def main():
    print("🧹 Clearing all demo data from surveillance system...")

    # Initialize database if needed
    init_db()

    # Clear all persons and detections
    result = clear_all_persons()

    if result['success']:
        print("✅ Successfully cleared all demo data!")
        print("📝 Database is now clean and ready for new data.")
    else:
        print(f"❌ Failed to clear demo data: {result['message']}")

if __name__ == "__main__":
    main()
