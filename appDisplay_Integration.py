import firebase_admin
from firebase_admin import credentials, db

# Path to your Firebase service account key
cred = credentials.Certificate("path/to/your-service-account.json")

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://<your-database-name>.firebaseio.com/'  # Replace with your DB URL
})

# Reference to your database
ref = db.reference('your/data/path')  # Example: 'users/user1'

# Data to send
data = {
    'name': 'Evan Hoffmann',
    'score': 95,
    'active': True
}

# Push data to the database (auto-generates a new key)
ref.push(data)

# Or set data at a specific path (overwrites existing data at that path)
# ref.set(data)
