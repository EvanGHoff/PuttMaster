import firebase_admin
from firebase_admin import credentials, firestore
import datetime

# Path to your Firebase service account key
cred = credentials.Certificate("Don't Upload\putt-master-firebase-firebase-adminsdk-fbsvc-da7f2a7f2b.json")

score = 99

app = firebase_admin.initialize_app(cred)

db = firestore.client()

miss_reason = {"Too Short", "Too Long", "Left", "Right"}


data = {
    'made': score==100,
    'missReason': "Right",
    'timestamp': datetime.datetime.now(tz=datetime.timezone.utc)
}


email = "jeramiegomez5@gmail.com"

email = email.replace("@", "_at_").replace(".", ",")

# Reference to the putts subcollection
putts_ref = db.collection("users").document(email).collection("putts")

# Get all documents in the subcollection
putts = putts_ref.stream()

n = sum(1 for _ in putts) + 1

db.collection("users").document(email).collection("putts").document(f"putt{n}").set(data)



print("Finished Running")
