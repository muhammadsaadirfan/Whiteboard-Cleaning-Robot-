# Whiteboard Cleaning Robot Control System

---

![Whiteboard Cleaning Robot](../White%20Board%20Cleaning%20Robot.png)

---

This project is a comprehensive control system for a whiteboard cleaning robot that uses computer vision to detect handwritten content on a whiteboard and autonomously erase it. The system includes a GUI for monitoring and controlling the robot, Bluetooth connectivity for communication, and object detection using YOLOv8.

## Features

- **Computer Vision System**:
  - Whiteboard detection using YOLOv8 model
  - Handwritten text detection and localization
  - ArUco marker detection for robot positioning
  - Grid-based path planning for efficient cleaning

- **Robot Control**:
  - Bluetooth communication with the robot
  - Manual and autonomous control modes
  - Emergency stop functionality
  - BLDC motor speed control

- **User Interface**:
  - Real-time camera feed display
  - Grid visualization of cleaning progress
  - Status indicators for system components
  - Image selection from Google Drive synced folder

- **Additional Features**:
  - Mobile image upload via QR code
  - Google Drive synchronization for whiteboard images
  - Multiple input modes (camera, video, image)

## System Requirements

### Hardware Requirements
- Linux-based system (tested on Ubuntu)
- Webcam or other video input device
- Bluetooth adapter for robot communication
- Whiteboard cleaning robot with Bluetooth capability
- ArUco markers for robot positioning

### Software Requirements
- Python 3.8 or higher
- pip package manager

## Installation Guide

### 1. Open Repository
```bash
cd whiteboard-cleaning-robot/ES_Project
```

### 2. Install Dependencies
Install the required Python packages using pip:

```bash
pip install opencv-python numpy pillow ultralytics pybluez pyserial gdown tkinter
```

### 3. System Dependencies
Install the following system packages:

```bash
sudo apt-get update
sudo apt-get install -y bluetooth bluez python3-tk
```

### 4. Bluetooth Configuration
Ensure your Bluetooth adapter is properly configured:

```bash
sudo systemctl start bluetooth
sudo systemctl enable bluetooth
```

### 5. Download YOLO Model
The system uses a pre-trained YOLOv8 model for whiteboard and text detection. The model file (`best.pt`) should be placed in:

```
ES_Project/yolo8_for_whiteboard_and_text_detection/best.pt
```

If the model file is not present, the system will attempt to download it automatically when first run.

### 6. Google Drive Setup (Optional)
The system can sync with a Google Drive folder containing whiteboard images. To enable this feature:

1. Create a Google Drive folder and share it publicly
2. Update the `GOOGLE_DRIVE_FOLDER_URL` variable in `main.py` with your folder URL
3. The first sync will happen automatically when you click the "Sync Drive" button in the GUI

## Running the Application

To start the application, run:

```bash
python3 main.py
```

## Usage Guide

### GUI Overview
The application provides a comprehensive GUI with the following components:

1. **Video Feed Display**:
   - Shows the live camera feed or selected image/video
   - Displays detected whiteboard and handwritten content

2. **Grid Display**:
   - Visual representation of the whiteboard grid
   - Shows dirty cells (black), robot position (yellow), and planned path (red arrows)

3. **Control Panel**:
   - Mode selection (Camera/Video/Image)
   - Camera ID/Video path/Image path configuration
   - Bluetooth device connection
   - Grid size configuration
   - BLDC motor speed control

4. **Action Buttons**:
   - Start Feed: Begin video feed from selected source
   - Connect Robot: Establish Bluetooth connection with the robot
   - Detect Board: Manually trigger whiteboard detection
   - Update Grid: Apply new grid dimensions
   - Sync Drive: Sync with Google Drive folder
   - START: Begin autonomous cleaning
   - Erase Board: Erase the entire whiteboard
   - Emergency Stop: Immediately stop all robot movement

### Basic Workflow

1. **Connect the Camera**:
   - Select "Camera" mode
   - Enter your camera ID (usually 0 or 2 for built-in/external cameras)
   - Click "Start Feed"

2. **Connect the Robot**:
   - Enter your robot's Bluetooth device name
   - Click "Connect Robot"

3. **Detect the Whiteboard**:
   - Position the camera to view the whiteboard
   - Click "Detect Board" or wait for automatic detection

4. **Configure the Grid**:
   - Adjust rows and columns as needed (default is 16x32)
   - Click "Update Grid"

5. **Start Cleaning**:
   - For specific dirty areas: Click "START"
   - For full whiteboard erase: Click "Erase Board"

### Mobile Image Upload

1. Click on the QR code displayed in the GUI
2. Scan the QR code with your mobile device
3. Upload images of the whiteboard
4. Click "Sync Drive" to download new images
5. Select an image from the list to analyze it

## Configuration Options

You can modify several parameters in the `main.py` file:

- `BAUD_RATE`: Bluetooth communication baud rate (default: 115200)
- `RFCOMM_PORT`: Bluetooth communication port (default: "/dev/rfcomm0")
- `SCAN_DURATION`: Bluetooth device scan duration in seconds (default: 10)
- `MAX_RETRIES`: Maximum number of Bluetooth connection attempts (default: 3)
- `GOOGLE_DRIVE_FOLDER_URL`: URL of the Google Drive folder for image sync
- `LOCAL_IMAGE_FOLDER`: Local folder for storing whiteboard images
- `QR_CODE_PATH`: Path to the QR code image for mobile upload

## Troubleshooting

### Bluetooth Connection Issues
1. Ensure Bluetooth is enabled on both the computer and robot
2. Verify the correct device name is entered
3. Check that the robot is powered on and in pairing mode
4. Try restarting the Bluetooth service:
   ```bash
   sudo systemctl restart bluetooth
   ```

### Camera Issues
1. Verify the correct camera ID is entered
2. Check that the camera is properly connected
3. Test the camera with other applications to ensure it works
4. For USB cameras, try different USB ports

### Object Detection Issues
1. Ensure proper lighting conditions
2. Verify the whiteboard is clearly visible in the camera frame
3. Check that the YOLO model file (`best.pt`) is in the correct location
4. Adjust the confidence threshold in the detection code if needed

### Performance Issues
1. For slower computers, reduce the camera resolution
2. Close other resource-intensive applications
3. Consider using a lower-resolution video source

## Project Structure

```
ES_Project/
├── assets/                  # Static assets
│   └── QR_gd.png            # QR code for mobile upload
├── whiteboard_images/       # Synced whiteboard images
├── yolo8_for_whiteboard_and_text_detection/
│   ├── best.pt              # YOLOv8 model weights
│   ├── model_run_webcam.py  # Webcam detection script
│   ├── model_test.py        # Model testing script
│   ├── model_train.py       # Model training script
│   └── runs/                # Training runs
├── main.py                  # Main application file
└── readme.md                # Project documentation
```

## Acknowledgments

- Ultralytics for the YOLOv8 object detection framework
- OpenCV for computer vision functionality
- The Python community for various supporting libraries
