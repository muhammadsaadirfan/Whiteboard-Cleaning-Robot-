#include <ESP32Servo.h>
#include "BluetoothSerial.h"

BluetoothSerial SerialBT;

Servo esc;
const int escPin = 13;
int motorSpeed = 1000;  // ESC minimum (stopped)

// ========== NEW: Individual PWM Speed for Each Motor ==========
int leftMotorSpeed = 255;   // Motor1 (Left) - default full speed
int rightMotorSpeed = 195;  // Motor2 (Right) - default full speed

// Optional: Bluetooth se change karne ke liye temporary variables
int tempLeftSpeed = 255;
int tempRightSpeed = 255;

// DC Motors Pins
#define MOTOR1_PWM 14   // Left Motor PWM
#define MOTOR1_FWD 26
#define MOTOR1_BWD 25
#define MOTOR2_PWM 27   // Right Motor PWM
#define MOTOR2_FWD 32
#define MOTOR2_BWD 33

// LEDs
#define LED_FORWARD 2
#define LED_BACKWARD 4
#define LED_LEFT 16
#define LED_RIGHT 17

void setup() {
  SerialBT.begin("Hybrid_Robot");

  esc.attach(escPin, 1000, 2000);
  esc.writeMicroseconds(1000);
  delay(3000);

  pinMode(MOTOR1_PWM, OUTPUT); pinMode(MOTOR1_FWD, OUTPUT); pinMode(MOTOR1_BWD, OUTPUT);
  pinMode(MOTOR2_PWM, OUTPUT); pinMode(MOTOR2_FWD, OUTPUT); pinMode(MOTOR2_BWD, OUTPUT);
  pinMode(LED_FORWARD, OUTPUT); pinMode(LED_BACKWARD, OUTPUT);
  pinMode(LED_LEFT, OUTPUT); pinMode(LED_RIGHT, OUTPUT);

  stopCar();
}

void loop() {
  if (SerialBT.available()) {
    String input = SerialBT.readStringUntil('\n');
    input.trim();

    if (input.length() == 0) return;

    char firstChar = input.charAt(0);

    // Car control (F, B, L, R, S)
    if (isAlpha(firstChar)) {
      switch (firstChar) {
        case 'F':
        case 'f': moveForward(); SerialBT.println("Forward"); break;
        case 'B':
        case 'b': moveBackward(); SerialBT.println("Backward"); break;
        case 'L':
        case 'l': turnLeft(); SerialBT.println("Left"); break;
        case 'R':
        case 'r': turnRight(); SerialBT.println("Right"); break;
        case 'S':
        case 's': stopCar(); SerialBT.println("Stop"); break;
        default:
          SerialBT.println("Invalid command");
      }
    }
    // ESC speed (1000-2000)
    else if (isDigit(firstChar)) {
      int newSpeed = input.toInt();
      if (newSpeed >= 1000 && newSpeed <= 2000) {
        motorSpeed = newSpeed;
        esc.writeMicroseconds(motorSpeed);
        SerialBT.print("ESC Speed: ");
        SerialBT.println(motorSpeed);
      } else {
        SerialBT.println("Invalid speed! Use 1000-2000");
      }
    }
    // NEW: Individual motor speed control via Bluetooth
    // Format: L200 (Left motor speed 200), R150 (Right motor speed 150)
    else if (firstChar == 'L' || firstChar == 'l') {
      String speedStr = input.substring(1);
      int newSpeed = speedStr.toInt();
      if (newSpeed >= 0 && newSpeed <= 255) {
        leftMotorSpeed = newSpeed;
        SerialBT.print("Left Motor Speed: ");
        SerialBT.println(leftMotorSpeed);
      } else {
        SerialBT.println("Invalid! Use L0 to L255");
      }
    }
    else if (firstChar == 'R' || firstChar == 'r') {
      String speedStr = input.substring(1);
      int newSpeed = speedStr.toInt();
      if (newSpeed >= 0 && newSpeed <= 255) {
        rightMotorSpeed = newSpeed;
        SerialBT.print("Right Motor Speed: ");
        SerialBT.println(rightMotorSpeed);
      } else {
        SerialBT.println("Invalid! Use R0 to R255");
      }
    }
    else {
      SerialBT.println("Unknown input");
    }
  }

  esc.writeMicroseconds(motorSpeed);
  delay(50);
}

// ==================== Car Control Functions (Individual Speeds) ====================
void allLEDsOff() {
  digitalWrite(LED_FORWARD, LOW);
  digitalWrite(LED_BACKWARD, LOW);
  digitalWrite(LED_LEFT, LOW);
  digitalWrite(LED_RIGHT, LOW);
}

void stopCar() {
  analogWrite(MOTOR1_PWM, 0);
  analogWrite(MOTOR2_PWM, 0);
  digitalWrite(MOTOR1_FWD, LOW); digitalWrite(MOTOR1_BWD, LOW);
  digitalWrite(MOTOR2_FWD, LOW); digitalWrite(MOTOR2_BWD, LOW);
  allLEDsOff();
}

void moveForward() {
  digitalWrite(MOTOR1_FWD, HIGH); digitalWrite(MOTOR1_BWD, LOW);
  digitalWrite(MOTOR2_FWD, HIGH); digitalWrite(MOTOR2_BWD, LOW);
  
  analogWrite(MOTOR1_PWM, leftMotorSpeed);   // Left motor individual speed
  analogWrite(MOTOR2_PWM, rightMotorSpeed);  // Right motor individual speed
  
  allLEDsOff();
  digitalWrite(LED_FORWARD, HIGH);
}

void moveBackward() {
  digitalWrite(MOTOR1_FWD, LOW); digitalWrite(MOTOR1_BWD, HIGH);
  digitalWrite(MOTOR2_FWD, LOW); digitalWrite(MOTOR2_BWD, HIGH);
  
  analogWrite(MOTOR1_PWM, leftMotorSpeed);
  analogWrite(MOTOR2_PWM, rightMotorSpeed);
  
  allLEDsOff();
  digitalWrite(LED_BACKWARD, HIGH);
}

void turnLeft() {
  digitalWrite(MOTOR1_FWD, LOW); digitalWrite(MOTOR1_BWD, HIGH);   // Left backward
  digitalWrite(MOTOR2_FWD, HIGH); digitalWrite(MOTOR2_BWD, LOW);  // Right forward
  
  analogWrite(MOTOR1_PWM, leftMotorSpeed);
  analogWrite(MOTOR2_PWM, rightMotorSpeed);
  
  allLEDsOff();
  digitalWrite(LED_LEFT, HIGH);
}

void turnRight() {
  digitalWrite(MOTOR1_FWD, HIGH); digitalWrite(MOTOR1_BWD, LOW);  // Left forward
  digitalWrite(MOTOR2_FWD, LOW); digitalWrite(MOTOR2_BWD, HIGH);  // Right backward
  
  analogWrite(MOTOR1_PWM, leftMotorSpeed);
  analogWrite(MOTOR2_PWM, rightMotorSpeed);
  
  allLEDsOff();
  digitalWrite(LED_RIGHT, HIGH);
}

