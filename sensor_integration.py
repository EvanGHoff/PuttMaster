import requests
import time
import csv
import numpy as np
from scipy import integrate
from scipy.spatial.transform import Rotation as R
from threading import Thread, Event
from collections import deque

esp32_url = "http://192.168.220.91"  # Change to match your setup
csv_filename = "sensor_log.csv"
sampling_rate_hz = 1000  # 1000 Hz
dt = 1.0 / sampling_rate_hz
stop_event = Event()
sensor_data_buffer = deque()

def sensor_reader(sensor_data_buffer, stop_event):
    while not stop_event.is_set():
        try:
            response = requests.get(esp32_url, timeout=0.5)
            if response.status_code == 200:
                raw_data = response.text.strip()
                try:
                    accel_line, gyro_line = raw_data.split('\n')
                    accel_vals = {k.strip(): float(v.strip()) for k, v in 
                                  (item.split(':') for item in accel_line.split('|'))}
                    gyro_vals = {k.strip(): float(v.strip().replace('dps', '')) for k, v in 
                                 (item.split(':') for item in gyro_line.split('|'))}

                    # Convert raw units
                    ax = accel_vals['Accel X'] / 256 * 9.8
                    ay = accel_vals['Accel Y'] / 256 * 9.8
                    az = accel_vals['Accel Z'] / 256 * 9.8
                    gx = gyro_vals['Gyro X']

                    timestamp = time.time()
                    sensor_data_buffer.append((timestamp, ax, ay, az, gx))
                except Exception as e:
                    print(f"[ERROR] Parse failed: {e} | Raw: {raw_data}")
        except Exception as e:
            print(f"[ERROR] Request failed: {e}")
        time.sleep(dt)

def data_logger(sensor_data_buffer, stop_event):
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'ax', 'ay', 'az', 'gx'])
        while not stop_event.is_set():
            if sensor_data_buffer:
                row = sensor_data_buffer.popleft()
                writer.writerow(row)
                f.flush()

def process_logged_data():
    print("\n[INFO] Processing logged data...")
    timestamps, ax, ay, az, gx = [], [], [], [], []

    with open(csv_filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row['timestamp']))
            ax.append(float(row['ax']))
            ay.append(float(row['ay']))
            az.append(float(row['az']))
            gx.append(float(row['gx']))

    ax = np.array(ax)
    ay = np.array(ay)
    az = np.array(az)
    gx = np.array(gx)
    timestamps = np.array(timestamps)

    dts = np.diff(timestamps, prepend=timestamps[0])
    mean_dt = np.mean(dts)

    # Use first second of data as baseline offset (stationary)
    N_offset = int(1.0 / mean_dt)
    ax0, ay0, az0 = np.mean(ax[:N_offset]), np.mean(ay[:N_offset]), np.mean(az[:N_offset])
    gx0 = np.mean(gx[:N_offset])

    # Remove offset
    ax -= ax0
    ay -= ay0
    az -= az0
    gx -= gx0

    # Integrate gyro to get pitch (degrees)
    pitch = integrate.cumulative_trapezoid(gx, dx=mean_dt, initial=0)

    # Rotate acceleration vector to remove gravity
    ax_nog = []
    for i in range(len(ax)):
        rot = R.from_rotvec([-np.deg2rad(pitch[i]), 0, 0])  # only pitch rotation
        g = rot.apply([ax[i], ay[i], az[i]])
        ax_nog.append(g[0])  # horizontal X-direction acceleration

    ax_nog = np.array(ax_nog)

    # Integrate to get velocity (in X)
    vx = integrate.cumulative_trapezoid(ax_nog, dx=mean_dt, initial=0)

    print(f"[RESULT] Final velocity: {vx[-1]:.2f} m/s")
    print(f"[RESULT] Final facing angle (pitch): {pitch[-1]:.2f} degrees")

# ---- Main ----
if __name__ == "__main__":
    try:
        reader_thread = Thread(target=sensor_reader, args=(sensor_data_buffer, stop_event))
        logger_thread = Thread(target=data_logger, args=(sensor_data_buffer, stop_event))
        reader_thread.start()
        logger_thread.start()

        print("[INFO] Logging... Press Ctrl+C to stop and process data.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt detected. Stopping threads...")
        stop_event.set()
        reader_thread.join()
        logger_thread.join()

        process_logged_data()
