import sqlite3
from datetime import datetime

DB_NAME = 'golf_swing.db'

def create_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS swing_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        swing_speed REAL NOT NULL,
        facing_angle REAL NOT NULL
    );
    ''')
    conn.commit()
    conn.close()
    print("Database and table created successfully.")

def insert_swing_data(swing_speed, facing_angle):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
    INSERT INTO swing_data (timestamp, swing_speed, facing_angle)
    VALUES (?, ?, ?)
    ''', (timestamp, swing_speed, facing_angle))
    conn.commit()
    conn.close()
    print(f"Swing data saved: Speed = {swing_speed:.2f} m/s, Angle = {facing_angle:.2f}°")

def fetch_all_swing_data():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM swing_data')
    data = cursor.fetchall()
    conn.close()
    return data
