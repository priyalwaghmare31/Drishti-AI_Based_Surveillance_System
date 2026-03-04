#!/usr/bin/env python3
"""
One-time cleanup script to delete all old demo records and verify the database is empty.
Run this script once to clear all existing data before adding your own.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import clear_all_persons, get_all_people

def main():
    print("🧹 Starting database cleanup...")

    # Clear all persons
    result = clear_all_persons()
    if result['success']:
        print("✅ All persons cleared successfully")
    else:
        print(f"❌ Error clearing persons: {result['message']}")
        return

    # Verify database is empty
    people = get_all_people()
    if len(people) == 0:
        print("✅ Database is now empty - ready for new data")
    else:
        print(f"❌ Database still contains {len(people)} records")

    print("🎉 Cleanup complete!")

if __name__ == "__main__":
    main()
