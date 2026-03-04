#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__) + '/src')
from database import add_person, get_all_people, init_db
import os

# Initialize database
init_db()

# Test adding a person
print('Testing add_person function...')

class MockFile:
    def __init__(self, filename, data):
        self.filename = filename
        self.data = data

    def save(self, path):
        with open(path, 'wb') as f:
            f.write(self.data)

mock_image = MockFile('test.jpg', b'fake image data')

# Test adding person
result = add_person(
    name='Test User',
    enroll='12345',
    branch='CSE',
    email='test@example.com',
    contact='1234567890',
    image_file=mock_image
)

print('Add person result:', result)

# Test getting all people
people = get_all_people()
print('All people:', people)

# Test duplicate name
result2 = add_person(
    name='Test User',
    enroll='67890',
    branch='IT',
    email='test2@example.com',
    contact='0987654321',
    image_file=mock_image
)

print('Duplicate name result:', result2)

print('Test completed successfully!')
