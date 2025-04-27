import requests

# Replace this with the actual IP address of your ESP32
esp32_ip = "http://192.168.220.91"  # Replace with the actual IP address of the ESP32
url = f"{esp32_ip}"  # If your ESP32 is serving data on the root

def get_sensor_data():
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Sensor Data Received:")
            print(response.text)
        else:
            print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from ESP32: {e}")

# Example usage
while True:
    get_sensor_data()
