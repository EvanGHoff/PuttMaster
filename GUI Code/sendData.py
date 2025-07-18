import firebase_admin
from firebase_admin import credentials, firestore
import datetime


def sendData(user, score, puttSpeed, puttPitch, missDir):

    db = firestore.client()

    miss_reason = {"Too Short", "Too Long", "Left", "Right"}


    data = {
        'made': score>=97,
        'missReason': missDir if score < 97 else "N/A",
        'speed': puttSpeed,
        'facing angle': puttPitch,
        'timestamp': datetime.datetime.now(tz=datetime.timezone.utc)
    }

    email = str(user).replace("@", "_at_").replace(".", ",")
    print(email)
    # Reference to the putts subcollection
    putts_ref = db.collection("users").document(email).collection("putts")

    # Get all documents in the subcollection
    putts = putts_ref.stream()

    n = sum(1 for _ in putts) + 1

    db.collection("users").document(email).collection("putts").document(f"putt{n}").set(data)

    print("Finished Running")
