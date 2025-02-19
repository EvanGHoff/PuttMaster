from flask import Flask, jsonify
import sqlite3

app = Flask(__name__)

@app.route('/data')
def get_data():
    conn = sqlite3.connect('/path/to/your/database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM your_table")
    rows = cursor.fetchall()
    conn.close()
    return jsonify(rows)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
