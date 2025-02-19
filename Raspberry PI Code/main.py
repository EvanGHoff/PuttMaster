# main.py
from bluetooth_receiver import connect_bluetooth_sensor
from database import create_database
from database import insert_swing_data
import asyncio

def main():
    create_database()
    print("Ready to receive swing data from Bluetooth sensor.")
    # asyncio.run(connect_bluetooth_sensor())
    insert_swing_data(123, 321)   # test data

if __name__ == "__main__":
    main()