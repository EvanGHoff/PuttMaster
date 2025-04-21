

// I2C device class (I2Cdev) demonstration Arduino sketch for ADXL345 class
// 10/7/2011 by Jeff Rowberg <jeff@rowberg.net>
// Updates should (hopefully) always be available at https://github.com/jrowberg/i2cdevlib
//
// Changelog:
//     2011-10-07 - initial release

/* ============================================
I2Cdev device library code is placed under the MIT license
Copyright (c) 2011 Jeff Rowberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
===============================================
*/

// Arduino Wire library is required if I2Cdev I2CDEV_ARDUINO_WIRE implementation
// is used in I2Cdev.h
#include "Wire.h"

// I2Cdev and ADXL345 must be installed as libraries, or else the .cpp/.h files
// for both classes must be in the include path of your project
#include "I2Cdev.h"
#include "ADXL345.h"
#include <Arduino.h>

#define IAM20380_ADDR 0x68

// Register addresses
#define WHO_AM_I       0x75
#define PWR_MGMT_1     0x6B
#define CONFIG         0x1A
#define GYRO_CONFIG    0x1B
#define GYRO_XOUT_H    0x43
#define GYRO_YOUT_H    0x45
#define GYRO_ZOUT_H    0x47

void I2C_writeRegister(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(IAM20380_ADDR);
    Wire.write(reg);
    Wire.write(value);
    Wire.endTransmission();
}

uint8_t I2C_readRegister(uint8_t reg) {
    Wire.beginTransmission(IAM20380_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);
    Wire.requestFrom(IAM20380_ADDR, 1);
    return Wire.available() ? Wire.read() : 0xFF;
}

void IAM_init() {
    Serial.println("Initializing IAM-20380...");
    I2C_writeRegister(PWR_MGMT_1, 0x09); // Select best clock source
    delay(100);
    I2C_writeRegister(GYRO_CONFIG, 0x08); // Set gyroscope full scale to ±500 dps
    I2C_writeRegister(CONFIG, 0x03); // Enable DLPF at ~44Hz for noise reduction
    Serial.println("IAM-20380 Initialized.");
}

void IAM_readGyro(float *gx, float *gy, float *gz) {
    Wire.beginTransmission(IAM20380_ADDR);
    Wire.write(GYRO_XOUT_H);
    Wire.endTransmission(false);
    Wire.requestFrom(IAM20380_ADDR, 6);

    if (Wire.available() == 6) {
        int16_t raw_gx = (Wire.read() << 8) | Wire.read();
        int16_t raw_gy = (Wire.read() << 8) | Wire.read();
        int16_t raw_gz = (Wire.read() << 8) | Wire.read();
        
        float sensitivity = 65.5;  // Sensitivity for ±500 dps
        *gx = raw_gx / sensitivity;
        *gy = raw_gy / sensitivity;
        *gz = raw_gz / sensitivity;
    } else {
        Serial.println("ERROR: Failed to read gyro data.");
    }
}

void runSelfTest() {
    Serial.println("Running self-test...");
    I2C_writeRegister(GYRO_CONFIG, 0xE0); // Enable self-test
    delay(200);
    float gx, gy, gz;
    IAM_readGyro(&gx, &gy, &gz);
    Serial.print("Self-test readings -> X: "); Serial.print(gx);
    Serial.print(" | Y: "); Serial.print(gy);
    Serial.print(" | Z: "); Serial.println(gz);
    I2C_writeRegister(GYRO_CONFIG, 0x08); // Restore normal operation
    Serial.println("Self-test complete.");
}
void resetIAM20380() {
    Wire.beginTransmission(0x68); // Sensor address
    Wire.write(0x6B); // PWR_MGMT_1 Register
    Wire.write(0x80); // Reset command
    Wire.endTransmission();
    delay(100);  // Wait for reset to complete
}
void wakeUpIAM20380() {
    Wire.beginTransmission(0x68);
    Wire.write(0x6B);
    Wire.write(0x00);  // Wake up command
    Wire.endTransmission();
    delay(10);  // Small delay for stability
}

// class default I2C address is 0x53
// specific I2C addresses may be passed as a parameter here
// ALT low = 0x53 (default for SparkFun 6DOF board)
// ALT high = 0x1D
ADXL345 accel;

int16_t ax, ay, az;



void setup() {
    // join I2C bus (I2Cdev library doesn't do this automatically)
    Wire.begin(21, 22);

    // initialize serial communication
    // (38400 chosen because it works as well at 8MHz as it does at 16MHz, but
    // it's really up to you depending on your project)
    Serial.begin(115200);

    // initialize device
    Serial.println("Initializing I2C devices...");
    accel.initialize();

    // verify connection
    Serial.println("Testing device connections...");
    Serial.println(accel.testConnection() ? "ADXL345 connection successful" : "ADXL345 connection failed");
    
    byte error;

    Serial.println("Trying address 0x53...");
    Wire.beginTransmission(0x53);
    error = Wire.endTransmission();
    if (error == 0) Serial.println(" ADXL345 found at 0x53!");
    else Serial.println(" No device at 0x53.");



  
    Wire.setClock(400000);
    resetIAM20380();
    wakeUpIAM20380();
    
    Serial.println("I2C Scanner: Scanning for devices...");
    for (uint8_t address = 1; address < 127; address++) {
        Wire.beginTransmission(address);
        if (Wire.endTransmission() == 0) {
            Serial.print("Device found at 0x");
            Serial.println(address, HEX);
        }
    }
    Serial.println("Scan complete.");
    
    Serial.println("Reading WHO_AM_I register...");
    uint8_t who_am_i = I2C_readRegister(WHO_AM_I);
    Serial.print("WHO_AM_I: 0x"); Serial.println(who_am_i, HEX);
    if (who_am_i != 0xB5) {
        Serial.println("ERROR: IAM-20380 not detected!");
        while (1);
    }
    
    IAM_init();
    runSelfTest();
   
}

void loop() {
    // read raw accel measurements from device
    accel.getAcceleration(&ax, &ay, &az);

    // display tab-separated accel x/y/z values
    Serial.print("accel:\t");
    Serial.print(ax); Serial.print("\t");
    Serial.print(ay); Serial.print("\t");
    Serial.println(az);


    float gx, gy, gz;
    IAM_readGyro(&gx, &gy, &gz);

    Serial.print("Gyro X: "); Serial.print(gx); Serial.print(" dps");
    Serial.print(" | Gyro Y: "); Serial.print(gy); Serial.print(" dps");
    Serial.print(" | Gyro Z: "); Serial.println(gz); Serial.print(" dps\n");

    delay(500);

   
}

