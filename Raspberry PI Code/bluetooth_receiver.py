# bluetooth_receiver.py
from bleak import BleakClient, BleakScanner
import asyncio
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from database import insert_swing_data

SENSOR_UUID = "<YOUR_SENSOR_CHARACTERISTIC_UUID>"  # Replace with your sensor's UUID

async def process_sensor_data(data):
    """Process received sensor data and insert into database."""
    accel_x = []
    gyro_z = []
    timestamps = []

    lines = data.decode('utf-8').strip().split("\n")
    for line in lines:
        if line == "END":
            if accel_x and gyro_z and timestamps:
                velocity = cumtrapz(np.array(accel_x), np.array(timestamps), initial=0)
                swing_speed = max(velocity)

                gyro_z_rad = np.radians(gyro_z)
                angle_z = np.degrees(cumtrapz(gyro_z_rad, timestamps, initial=0))
                facing_angle = angle_z[-1]

                insert_swing_data(swing_speed, facing_angle)
                print(f"Recorded Swing - Speed: {swing_speed:.2f} m/s, Angle: {facing_angle:.2f}°")
            accel_x.clear()
            gyro_z.clear()
            timestamps.clear()
        else:
            accel_value, gyro_value, timestamp = map(float, line.split(","))
            accel_x.append(accel_value)
            gyro_z.append(gyro_value)
            timestamps.append(timestamp)

async def connect_bluetooth_sensor():
    """Connect to the BLE sensor and handle incoming data."""
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover()
    for i, device in enumerate(devices):
        print(f"{i}: {device.name} - {device.address}")

    index = int(input("Select the device index to connect: "))
    address = devices[index].address

    async with BleakClient(address) as client:
        print(f"Connected to {devices[index].name}")

        def handle_data(_, data):
            asyncio.create_task(process_sensor_data(data))

        await client.start_notify(SENSOR_UUID, handle_data)
        print("Receiving data... Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await client.stop_notify(SENSOR_UUID)
            print("Stopped receiving data.")

if __name__ == "__main__":
    asyncio.run(connect_bluetooth_sensor())
