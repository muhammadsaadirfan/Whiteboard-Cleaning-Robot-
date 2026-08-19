import bluetooth
import serial
import time
import subprocess
import os
import cv2
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
import cv2.aruco as aruco
from collections import defaultdict
import heapq
from http.server import SimpleHTTPRequestHandler, HTTPServer
import socket
import webbrowser
import json
import gdown

GOOGLE_DRIVE_FOLDER_URL = 'https://drive.google.com/drive/folders/1wnmQXOJQ6o85BpR1K3R_byHu-h5bwEtw?usp=sharing'
LOCAL_IMAGE_FOLDER = 'ES_Project/whiteboard_images'
QR_CODE_PATH = 'ES_Project/assests/QR_gd.png'

if not os.path.exists(LOCAL_IMAGE_FOLDER):
    os.makedirs(LOCAL_IMAGE_FOLDER)

def sync_drive_to_local():
    print("Starting sync with Google Drive folder...")
    try:
        gdown.download_folder(
            url=GOOGLE_DRIVE_FOLDER_URL,
            output=LOCAL_IMAGE_FOLDER,
            quiet=False,
            remaining_ok=True
        )
        print(f"\nSync complete! Images updated in '{LOCAL_IMAGE_FOLDER}'")
    except Exception as e:
        print(f"Sync error: {e}")

BAUD_RATE = 115200
RFCOMM_PORT = "/dev/rfcomm0"
SCAN_DURATION = 10
MAX_RETRIES = 3

def find_bluetooth_device(device_name):
    print(f"Scanning for Bluetooth device: {device_name}")
    for attempt in range(MAX_RETRIES):
        try:
            devices = bluetooth.discover_devices(lookup_names=True, duration=SCAN_DURATION)
            for addr, name in devices:
                if name == device_name:
                    print(f"Found {device_name} with MAC address: {addr}")
                    return addr
            print(f"Attempt {attempt + 1}/{MAX_RETRIES}: Device {device_name} not found")
        except Exception as e:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES}: Error scanning for devices: {e}")
        time.sleep(2)
    return None

def bind_rfcomm(mac_address, rfcomm_num=0):
    try:
        if os.path.exists(f"/dev/rfcomm{rfcomm_num}"):
            print(f"Releasing existing /dev/rfcomm{rfcomm_num}")
            subprocess.run(["sudo", "rfcomm", "release", str(rfcomm_num)], check=True)
        print(f"Binding {mac_address} to /dev/rfcomm{rfcomm_num}")
        subprocess.run(["sudo", "rfcomm", "bind", str(rfcomm_num), mac_address, "1"], check=True)
        time.sleep(2)
        if os.path.exists(f"/dev/rfcomm{rfcomm_num}"):
            print(f"Successfully bound to /dev/rfcomm{rfcomm_num}")
            return f"/dev/rfcomm{rfcomm_num}"
        else:
            print(f"Failed to create /dev/rfcomm{rfcomm_num}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error binding rfcomm: {e}")
        return None

def connect_bluetooth(port):
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print(f"Connected to {port}")
        return ser
    except serial.SerialException as e:
        print(f"Error connecting to {port}: {e}")
        return None

def send_command(ser, command):
    if ser and ser.is_open:
        try:
            ser.write(command.encode())
            print(f"Sent command: {command}")
            return True
        except serial.SerialException as e:
            print(f"Error sending command: {e}")
            return False
    else:
        print("Serial port not connected")
        return False

class CameraFeed:
    def __init__(self, source='camera', camera_id=2, video_path=None, image_path=None):
        self.source = source
        self.camera_id = camera_id
        self.video_path = video_path
        self.image_path = image_path
        self.cap = None
        self.running = False
        self.current_frame = None
        self.grid_image = None
        self.whiteboard_detected = False
        self.whiteboard_bbox = None
        self.whiteboard_h = 0
        self.whiteboard_w = 0
        self.dirty_cells = set()
        self.robot_pose = None
        self.planned_path = []
        self.lock = threading.Lock()
        self.model = YOLO("ES_Project/yolo8_for_whiteboard_and_text_detection/best.pt")
        self.num_rows = 16
        self.num_cols = 32
        self.border = 4

        # ArUco setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict)
        self.tag_size = 0.06  # 6cm - adjust to your printed tag size
        self.camera_params = [600.0, 600.0, 600.0, 250.0]  # fx, fy, cx, cy - updated for 1200x450 frame
        self.camera_matrix = np.array([[self.camera_params[0], 0, self.camera_params[2]],
                                       [0, self.camera_params[1], self.camera_params[3]],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(5, dtype=np.float32)

        if self.source == 'video' and self.video_path:
            self.out = cv2.VideoWriter(
                "/home/muhammad/Books/5th semester/ES2/ES2_project/ES_Project/yolo8_for_whiteboard_and_text_detection/output_detected.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,
                (1200, 450)
            )
        else:
            self.out = None

    def start(self):
        if self.source == 'camera':
            self.cap = cv2.VideoCapture(self.camera_id)
        elif self.source == 'video':
            self.cap = cv2.VideoCapture(self.video_path)
        elif self.source == 'image':
            self.cap = None
        if self.source != 'image' and not self.cap.isOpened():
            print(f"Warning: Could not open {'camera' if self.source == 'camera' else 'video'}")
            return False
        self.running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()
        return True
    
    def _update_frame(self):
        while self.running:
            if self.source == 'image':
                if not self.image_path:
                    break
                frame = cv2.imread(self.image_path)
                if frame is None:
                    break
                frame = cv2.resize(frame, (1200, 500))
                self.running = False
            else:
                ret, frame = self.cap.read()
                if not ret:
                    if self.source == 'video':
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                frame = cv2.resize(frame, (1200, 500))

            display_frame = frame.copy()

            if self.whiteboard_bbox:
                y1, y2, x1, x2 = self.whiteboard_bbox
                whiteboard_frame = frame[y1:y2, x1:x2]
                whiteboard_display = whiteboard_frame.copy()

                # ArUco Detection on whiteboard region
                gray = cv2.cvtColor(whiteboard_display, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

                if ids is not None:
                    for i in range(len(ids)):
                        if ids[i] == 5:  # Change ID to your robot's tag ID
                            pts = corners[i][0].astype(int)
                            cv2.polylines(whiteboard_display, [pts], True, (0, 255, 0), 5)  # Thick green box

                            # Estimate pose
                            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i:i+1], self.tag_size, self.camera_matrix, self.dist_coeffs)
                            rvec = rvecs[0][0]
                            tvec = tvecs[0][0]
                            rot_mat, _ = cv2.Rodrigues(rvec)

                            # Draw 3D coordinate axes
                            axis_len = self.tag_size
                            axis_3D = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,-axis_len]]).reshape(-1,3)
                            axis_2D, _ = cv2.projectPoints(axis_3D, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                            axis_2D = axis_2D.astype(int).reshape(-1,2)
                            origin = tuple(axis_2D[0])
                            cv2.line(whiteboard_display, origin, tuple(axis_2D[1]), (0, 0, 255), 3)   # X red
                            cv2.line(whiteboard_display, origin, tuple(axis_2D[2]), (0, 255, 0), 3)   # Y green
                            cv2.line(whiteboard_display, origin, tuple(axis_2D[3]), (255, 0, 0), 3)   # Z blue

                            # Display tag ID and position
                            pos_text = np.round(tvec, 3)
                            cv2.putText(whiteboard_display, f"ID {ids[i][0]} Pos {pos_text}",
                                        (pts[0][0], pts[0][1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                            # Calculate grid position and facing direction
                            center_x = int(np.mean(pts[:, 0]))
                            center_y = int(np.mean(pts[:, 1]))
                            cell_h = self.whiteboard_h / self.num_rows
                            cell_w = self.whiteboard_w / self.num_cols
                            row = int(center_y / cell_h)
                            col = int(center_x / cell_w)

                            # Facing direction from rotation matrix
                            direction_vec = rot_mat[:2, 0]  # X-axis direction
                            angle = np.arctan2(direction_vec[1], direction_vec[0])
                            facing = int((angle / (np.pi / 2) + 4.5) % 4)  # 0=up, 1=right, 2=down, 3=left

                            with self.lock:
                                old_pose = self.robot_pose
                                self.robot_pose = (row, col, facing)
                                if old_pose and old_pose[:2] != (row, col):
                                    # Robot moved - clean the previous 4x4 area
                                    prev_row, prev_col, _ = old_pose
                                    for dr in range(-1, 3):
                                        for dc in range(-1, 3):
                                            occ_row = prev_row + dr
                                            occ_col = prev_col + dc
                                            if 0 <= occ_row < self.num_rows and 0 <= occ_col < self.num_cols:
                                                cell = (occ_row, occ_col)
                                                self.dirty_cells.discard(cell)

                annotated_frame = cv2.resize(whiteboard_display, (1200, 500))
                with self.lock:
                    self.current_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    self.whiteboard_detected = True
                self.compute_grid_image()
            else:
                with self.lock:
                    self.current_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    self.whiteboard_detected = False
                    self.robot_pose = None
            if self.out:
                self.out.write(frame)


    def compute_grid_image(self):
        grid_height = 450
        grid_width = 1200
        grid_image = np.full((grid_height, grid_width, 3), 255, np.uint8)
        cell_h = grid_height / self.num_rows
        cell_w = grid_width / self.num_cols
        
        # Draw border in red
        border_color = (0, 0, 255)  # BGR red
        # Top and bottom borders
        for border_row in range(self.border):
            y_start = int(border_row * cell_h)
            y_end = int((border_row + 1) * cell_h)
            cv2.rectangle(grid_image, (0, y_start), (grid_width, y_end), border_color, -1)
            
            bottom_row = self.num_rows - 1 - border_row
            y_start = int(bottom_row * cell_h)
            y_end = int((bottom_row + 1) * cell_h)
            cv2.rectangle(grid_image, (0, y_start), (grid_width, y_end), border_color, -1)
        
        # Left and right borders
        for border_col in range(self.border):
            x_start = int(border_col * cell_w)
            x_end = int((border_col + 1) * cell_w)
            cv2.rectangle(grid_image, (x_start, 0), (x_end, grid_height), border_color, -1)
            
            right_col = self.num_cols - 1 - border_col
            x_start = int(right_col * cell_w)
            x_end = int((right_col + 1) * cell_w)
            cv2.rectangle(grid_image, (x_start, 0), (x_end, grid_height), border_color, -1)

        # Draw track color borders
        track_centers = [3, 7, 11]
        track_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # red, green, blue in BGR
        for i, center in enumerate(track_centers):
            start_row = center - 1
            end_row = center + 3
            y_start = int(start_row * cell_h)
            y_end = int(end_row * cell_h)
            cv2.rectangle(grid_image, (0, y_start), (grid_width, y_end), track_colors[i], 2)

        # Draw remaining dirty cells in black (only inner)
        for row, col in self.dirty_cells:
            if self.border <= row < self.num_rows - self.border and self.border <= col < self.num_cols - self.border:
                y_start = int(row * cell_h)
                y_end = int((row + 1) * cell_h)
                x_start = int(col * cell_w)
                x_end = int((col + 1) * cell_w)
                cv2.rectangle(grid_image, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)
        
        # Draw grid lines
        for r in range(self.num_rows + 1):
            y = int(r * cell_h)
            cv2.line(grid_image, (0, y), (grid_width, y), (200, 200, 200), 1)
        for c in range(self.num_cols + 1):
            x = int(c * cell_w)
            cv2.line(grid_image, (x, 0), (x, grid_height), (200, 200, 200), 1)
        
        # Draw planned path with arrows
        if self.planned_path:
            for i in range(len(self.planned_path) - 1):
                r1, c1 = self.planned_path[i]
                r2, c2 = self.planned_path[i + 1]
                center1_x = int((c1 + 0.5) * cell_w)
                center1_y = int((r1 + 0.5) * cell_h)
                center2_x = int((c2 + 0.5) * cell_w)
                center2_y = int((r2 + 0.5) * cell_h)
                cv2.arrowedLine(grid_image, (center1_x, center1_y), (center2_x, center2_y), (0, 0, 255), 1, tipLength=0.3)
        
        # Robot position: 4x4 cleaned area (already white since not in dirty_cells)
        # Show duster as yellow rectangle on current cell + arrow
        if self.robot_pose:
            row, col, facing = self.robot_pose
            
            # Current cell yellow (duster shape)
            y_start = int(row * cell_h)
            y_end = int((row + 1) * cell_h)
            x_start = int(col * cell_w)
            x_end = int((col + 1) * cell_w)
            cv2.rectangle(grid_image, (x_start, y_start), (x_end, y_end), (0, 255, 255), -1)
            
            # Direction arrow (red)
            center_x = x_start + (x_end - x_start) // 2
            center_y = y_start + (y_end - y_start) // 2
            arrow_len = min((x_end - x_start) // 2, (y_end - y_start) // 2)
            if facing == 0:      # up
                end_x, end_y = center_x, center_y - arrow_len
            elif facing == 1:    # right
                end_x, end_y = center_x + arrow_len, center_y
            elif facing == 2:    # down
                end_x, end_y = center_x, center_y + arrow_len
            elif facing == 3:    # left
                end_x, end_y = center_x - arrow_len, center_y
            cv2.arrowedLine(grid_image, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3, tipLength=0.5)
        
        with self.lock:
            self.grid_image = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)

    def get_frame(self):
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_grid_image(self):
        with self.lock:
            return self.grid_image.copy() if self.grid_image is not None else None
    
    def is_whiteboard_detected(self):
        with self.lock:
            return self.whiteboard_detected
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()

class GridManager:
    def __init__(self, rows=16, cols=32, width=1200, height=450):
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height
        self.cell_width = width / cols
        self.cell_height = height / rows
    
    def update_grid_size(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.cell_width = self.width / cols
        self.cell_height = self.height / rows

class StatusManager:
    def __init__(self):
        self.statuses = {
            'connected': False,
            'robot_detected': False,
            'whiteboard_detected': False
        }
        self.callbacks = []
    
    def set_status(self, status_name, value):
        if status_name in self.statuses:
            self.statuses[status_name] = value
            self._notify_callbacks()
    
    def get_status(self, status_name):
        return self.statuses.get(status_name, False)
    
    def register_callback(self, callback):
        self.callbacks.append(callback)
    
    def _notify_callbacks(self):
        for callback in self.callbacks:
            callback(self.statuses)

class RobotController:
    def __init__(self):
        self.position = {'x': 0, 'y': 0}
        self.is_running = False
        self.callbacks = []
        self.ser = None
        self.mac_address = None
        self.port = None
        self.confirm_event = threading.Event()
        self.confirm_result = False
        self.need_confirm = False
        self.confirm_message = ""
        
    def connect(self, device_name):
        self.mac_address = find_bluetooth_device(device_name)
        if not self.mac_address:
            print("Failed to find Bluetooth device")
            return False
        self.port = bind_rfcomm(self.mac_address)
        if not self.port:
            print("Failed to bind rfcomm port")
            return False
        self.ser = connect_bluetooth(self.port)
        if not self.ser:
            print("Failed to connect to Bluetooth device")
            return False
        print("Bluetooth connection established")
        return True

    def move_up(self):
        self.position['y'] -= 1
        print(f"Moving UP - Position: {self.position}")
        send_command(self.ser, 'F')
        self._notify_movement()
    
    def move_down(self):
        self.position['y'] += 1
        print(f"Moving DOWN - Position: {self.position}")
        send_command(self.ser, 'B')
        self._notify_movement()
    
    def move_left(self):
        self.position['x'] -= 1
        print(f"Moving LEFT - Position: {self.position}")
        send_command(self.ser, 'L')
        self._notify_movement()
    
    def move_right(self):
        self.position['x'] += 1
        print(f"Moving RIGHT - Position: {self.position}")
        send_command(self.ser, 'R')
        self._notify_movement()
    
    def start_cleaning(self):
        self.is_running = True
        print("Cleaning started")
        self._notify_movement()
    
    def stop_cleaning(self):
        self.is_running = False
        print("Cleaning stopped")
        send_command(self.ser, 'S')
        self._notify_movement()
    
    def register_callback(self, callback):
        self.callbacks.append(callback)
    
    def _notify_movement(self):
        for callback in self.callbacks:
            callback(self.position, self.is_running)
    
    def get_tsp_approx_order(self, dirty_cells, start):
        unvisited = set(dirty_cells)
        current = start
        order = []
        while unvisited:
            closest = min(unvisited, key=lambda p: abs(p[0] - current[0]) + abs(p[1] - current[1]))
            order.append(closest)
            unvisited.remove(closest)
            current = closest
        return order

    def get_coverage_order(self, rows, cols, border):
        order = []
        track_centers = [11, 7, 3]  # for 3 tracks: bottom to top
        for track_idx, center in enumerate(track_centers):
            if track_idx % 2 == 0:
                cols_range = range(border, cols - border + 1, 3)
            else:
                cols_range = range(cols - border - 1, border - 1, -3)
            for c in cols_range:
                order.append((center, c))
        return order

    def a_star_path(self, start, goal, rows, cols, border):
        def heuristic(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, []))
        visited = set()
        directions = [(-1, 0, 'up'), (1, 0, 'down'), (0, -1, 'left'), (0, 1, 'right')]
        
        while open_set:
            _, cost, current, path = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            if current == goal:
                return path
            for dr, dc, move in directions:
                nr, nc = current[0] + dr, current[1] + dc
                if border <= nr < rows - border and border <= nc < cols - border:
                    new_cost = cost + 1
                    new_path = path + [move]
                    priority = new_cost + heuristic((nr, nc), goal)
                    heapq.heappush(open_set, (priority, new_cost, (nr, nc), new_path))
        return []

    def align_and_move(self, move_dir, current_facing, ser):
        dir_map = {'up': 0, 'right': 1, 'down': 2, 'left': 3}
        target_dir = dir_map[move_dir]
        diff = (target_dir - current_facing) % 4
        if diff == 0:
            send_command(ser, 'F')
        elif diff == 1:
            send_command(ser, 'R')
            send_command(ser, 'F')
            current_facing = (current_facing + 1) % 4
        elif diff == 3:
            send_command(ser, 'L')
            send_command(ser, 'F')
            current_facing = (current_facing - 1) % 4
        elif diff == 2:
            send_command(ser, 'R')
            send_command(ser, 'R')
            send_command(ser, 'F')
            current_facing = (current_facing + 2) % 4
        return current_facing

    def align_to_facing(self, target_facing, current_facing, ser):
        diff = (target_facing - current_facing) % 4
        if diff == 0:
            pass
        elif diff == 1:
            send_command(ser, 'R')
            current_facing = (current_facing + 1) % 4
        elif diff == 2:
            send_command(ser, 'R')
            send_command(ser, 'R')
            current_facing = (current_facing + 2) % 4
        elif diff == 3:
            send_command(ser, 'L')
            current_facing = (current_facing - 1) % 4
        return current_facing

    
    def whiteboard_clean(self, initial_pose, rows, cols, camera):
        if not self.is_running:
            return

        if not self.ser or not self.ser.is_open:
            print("Bluetooth not connected!")
            return

        if not camera.robot_pose:
            print("Robot pose not detected yet! Cannot start.")
            return

        print("Starting hybrid erase sequence: Track 1 = Position-based, Rest = Time-based")

        track_rows = [11, 7, 3]  # نیچے سے اوپر: Track 1 (11), Track 2 (7), Track 3 (3)
        border = camera.border
        right_end_zone = cols - border - 6  # دائیں طرف آخری ~10 سیلز
        left_end_zone = border + 5         # بائیں طرف آخری ~10 سیلز

        u_turn_time = None  # پہلے U-turn کا وقت یہاں سٹور ہوگا

        def send(cmd, delay=0.3):
            if not self.is_running:
                return False
            send_command(self.ser, cmd)
            print(f"Sent: {cmd}")
            time.sleep(delay)
            return True

        def wait_for_robot():
            while camera.robot_pose is None and self.is_running:
                print("Robot temporarily lost, waiting...")
                time.sleep(0.5)
            if self.is_running:
                print("Robot visible again!")

        # ============== TRACK 1: POSITION-BASED (پہلا ٹریک) ==============
        print("Track 1: Position-based control (waiting to reach row ~11)")
        target_row = track_rows[0]

        # انتظار کرو جب تک روبوٹ پہلے ٹریک پر نہ آ جائے
        while self.is_running:
            if camera.robot_pose:
                row, col, facing = camera.robot_pose
                if abs(row - target_row) <= 1:
                    print(f"Robot on Track 1 (row {row}), starting movement")
                    break
            else:
                wait_for_robot()
            time.sleep(0.5)

        send('F')  # شروع کرو

        # چلتے رہو جب تک دائیں طرف آخری زون میں نہ پہنچ جائیں
        print("Moving right until end zone...")
        while self.is_running:
            if camera.robot_pose:
                row, col, facing = camera.robot_pose
                if col >= right_end_zone:
                    print(f"Reached right end (col {col})! Stopping for U-turn")
                    send('S')
                    time.sleep(1.0)
                    break
            else:
                print("Robot lost during Track 1, stopping temporarily")
                send('S')
                wait_for_robot()
                send('F')
            time.sleep(0.5)

        # ============== LEFT U-TURN (پہلا موڑ) ==============
        print("Performing Left U-turn (and measuring time)")
        u_turn_start = time.time()

        send('L', delay=1)
        send('S', delay=1.0)
        send('F', delay=0.5)   # تھوڑا آگے بڑھنے دو
        send('S', delay=1.0)
        send('L', delay=0.7)
        send('S', delay=1.0)
        send('F', delay=0.5)   # تھوڑا آگے بڑھنے دو
        send('S', delay=1.0)

        u_turn_time = time.time() - u_turn_start
        print(f"First U-turn completed in {u_turn_time:.2f} seconds. Will use this for remaining tracks.")

        # ============== TRACK 2 & 3: TIME-BASED (باقی ٹریکس) ==============
        track_move_time = 5.0  # ایک ٹریک چلنے کا تخمینی وقت (آپ ٹیسٹ کرکے ایڈجسٹ کر سکتے ہیں، مثلاً 32-38 سیکنڈ)

        for track_idx in range(1, 2):  # Track 2 اور Track 3
            if not self.is_running:
                break

            direction = "left" if track_idx == 1 else "right"
            print(f"Track {track_idx + 1}: Time-based movement ({direction}), duration ~{track_move_time}s")

            send('F')
            time.sleep(track_move_time)  # پورا ٹریک چلنے دو

            if track_idx == 2:  # آخری ٹریک — کوئی U-turn نہیں
                send('S')
                time.sleep(1.0)
                print("Last track completed. Full erase ES_Project!")
                break

            # اگلا ٹریک شروع
            send('F')

        # فائنل سٹاپ (سیفٹی کے لیے)
        send('S')
        time.sleep(1.0)
        print("Full board erase sequence completed (Hybrid mode)!")
        self.stop_cleaning()

    def auto_clean(self, dirty_cells, initial_pose, rows, cols, camera):
        if not self.is_running:
            return
        border = camera.border
        order = self.get_tsp_approx_order(dirty_cells, initial_pose[:2])
        current_pos = initial_pose[:2]
        current_facing = initial_pose[2]
        for nxt in order:
            while current_pos != nxt and self.is_running:
                path = self.a_star_path(current_pos, nxt, rows, cols, border)
                if not path:
                    break
                for m in path:
                    if not self.is_running:
                        return
                    current_facing = self.align_and_move(m, current_facing, self.ser)
                    time.sleep(1)
                    if camera.robot_pose:
                        current_pos = camera.robot_pose[:2]
                        current_facing = camera.robot_pose[2]
                    else:
                        if m == 'up':
                            current_pos = (current_pos[0] - 1, current_pos[1])
                        elif m == 'down':
                            current_pos = (current_pos[0] + 1, current_pos[1])
                        elif m == 'left':
                            current_pos = (current_pos[0], current_pos[1] - 1)
                        elif m == 'right':
                            current_pos = (current_pos[0], current_pos[1] + 1)
                    with camera.lock:
                        for dr in range(-1, 3):
                            for dc in range(-1, 3):
                                occ_row = current_pos[0] + dr
                                occ_col = current_pos[1] + dc
                                if border <= occ_row < rows - border and border <= occ_col < cols - border:
                                    cell = (occ_row, occ_col)
                                    camera.dirty_cells.discard(cell)
        start_pos = initial_pose[:2]
        while current_pos != start_pos and self.is_running:
            path = self.a_star_path(current_pos, start_pos, rows, cols, border)
            if not path:
                break
            for m in path:
                if not self.is_running:
                    return
                current_facing = self.align_and_move(m, current_facing, self.ser)
                time.sleep(1)
                if camera.robot_pose:
                    current_pos = camera.robot_pose[:2]
                    current_facing = camera.robot_pose[2]
                else:
                    if m == 'up':
                        current_pos = (current_pos[0] - 1, current_pos[1])
                    elif m == 'down':
                        current_pos = (current_pos[0] + 1, current_pos[1])
                    elif m == 'left':
                        current_pos = (current_pos[0], current_pos[1] - 1)
                    elif m == 'right':
                        current_pos = (current_pos[0], current_pos[1] + 1)
                with camera.lock:
                    for dr in range(-1, 3):
                        for dc in range(-1, 3):
                            occ_row = current_pos[0] + dr
                            occ_col = current_pos[1] + dc
                            if border <= occ_row < rows - border and border <= occ_col < cols - border:
                                cell = (occ_row, occ_col)
                                camera.dirty_cells.discard(cell)
        self.stop_cleaning()

    def cleanup(self):
        if self.ser and self.ser.is_open:
            send_command(self.ser, 'S')
            self.ser.close()
            print("Serial connection closed")
        try:
            subprocess.run(["sudo", "rfcomm", "release", "0"], check=True)
            print("Released /dev/rfcomm0")
        except subprocess.CalledProcessError as e:
            print(f"Error releasing rfcomm: {e}")

# باقی تمام کلاسز اور GUI کوڈ بالکل وہی ہے جو آپ نے دیا تھا — یہاں کاپی پیسٹ کر دیا گیا ہے

class StatusIndicator:
    def __init__(self, parent, text, x, y_pos):
        self.frame = tk.Frame(parent, bg='#000000')
        self.frame.place(x=x, y=y_pos)
        self.canvas = tk.Canvas(self.frame, width=40, height=40, 
                               bg='#000000', highlightthickness=0)
        self.canvas.pack(side=tk.LEFT)
        self.circle = self.canvas.create_oval(8, 8, 40, 40, 
                                             fill='#000000', outline='')
        self.check = None
        self.label = tk.Label(self.frame, text=text, fg='#D4D4D4', 
                            bg='#000000', font=('Arial', 17))
        self.label.pack(side=tk.LEFT, padx=(17, 0))
    
    def set_active(self, active):
        color = '#AAAAAA' if active else '#3C3C3C'
        self.canvas.itemconfig(self.circle, fill=color)
        if active and not self.check:
            self.check = self.canvas.create_text(30, 30, text="✓", 
                                                fill='#FFFFFF', 
                                                font=('Arial', 28, 'bold'))
        elif not active and self.check:
            self.canvas.delete(self.check)
            self.check = None

class DirectionalPad:
    def __init__(self, parent, x, y, controller):
        self.controller = controller
        self.canvas = tk.Canvas(parent, width=280, height=280,
                               bg='#000000', highlightthickness=0)
        self.canvas.place(x=x, y=y)
        self.canvas.create_oval(20, 20, 260, 260, fill='#3C3C3C',
                               outline='#AAAAAA', width=3)
        self.canvas.create_oval(90, 90, 190, 190, fill='#252526',
                               outline='#000000', width=3)
        self._create_direction_buttons()
    
    def _create_direction_buttons(self):
        up_btn = self.canvas.create_polygon(140, 50, 120, 85, 160, 85,
                                           fill='#D4D4D4', outline='',
                                           tags='up_btn')
        self.canvas.tag_bind('up_btn', '<Button-1>', 
                           lambda e: self.controller.move_up())
        self.canvas.tag_bind('up_btn', '<Enter>',
                           lambda e: self.canvas.itemconfig(up_btn, fill='#B0BEC5'))
        self.canvas.tag_bind('up_btn', '<Leave>',
                           lambda e: self.canvas.itemconfig(up_btn, fill='#D4D4D4'))
        down_btn = self.canvas.create_polygon(140, 230, 120, 195, 160, 195,
                                             fill='#D4D4D4', outline='',
                                             tags='down_btn')
        self.canvas.tag_bind('down_btn', '<Button-1>',
                           lambda e: self.controller.move_down())
        self.canvas.tag_bind('down_btn', '<Enter>',
                           lambda e: self.canvas.itemconfig(down_btn, fill='#B0BEC5'))
        self.canvas.tag_bind('down_btn', '<Leave>',
                           lambda e: self.canvas.itemconfig(down_btn, fill='#D4D4D4'))
        left_btn = self.canvas.create_polygon(50, 140, 85, 120, 85, 160,
                                             fill='#D4D4D4', outline='',
                                             tags='left_btn')
        self.canvas.tag_bind('left_btn', '<Button-1>',
                           lambda e: self.controller.move_left())
        self.canvas.tag_bind('left_btn', '<Enter>',
                           lambda e: self.canvas.itemconfig(left_btn, fill='#B0BEC5'))
        self.canvas.tag_bind('left_btn', '<Leave>',
                           lambda e: self.canvas.itemconfig(left_btn, fill='#D4D4D4'))
        right_btn = self.canvas.create_polygon(230, 140, 195, 120, 195, 160,
                                              fill='#D4D4D4', outline='',
                                              tags='right_btn')
        self.canvas.tag_bind('right_btn', '<Button-1>',
                           lambda e: self.controller.move_right())
        self.canvas.tag_bind('right_btn', '<Enter>',
                           lambda e: self.canvas.itemconfig(right_btn, fill='#B0BEC5'))
        self.canvas.tag_bind('right_btn', '<Leave>',
                           lambda e: self.canvas.itemconfig(right_btn, fill='#D4D4D4'))

class RobotControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Control System")
        self.root.configure(bg='#000000')
        self.root.geometry("1920x1080")
        self.mode = tk.StringVar(value="camera")
        self.camera = None
        self.grid_manager = GridManager()
        self.status_manager = StatusManager()
        self.robot_controller = RobotController()
        self.saved_dirty_cells = set()
        self._setup_ui()
        self.status_manager.register_callback(self._on_status_update)
        self.whiteboard_image_paths = []
        self._load_whiteboard_images()
        self._check_confirm()

    def _setup_ui(self):
        main_frame = tk.Frame(self.root, bg='#000000')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=(100, 30), pady=30)

        left_frame = tk.Frame(main_frame, bg='#000000')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_canvas = tk.Canvas(left_frame, width=1200, height=450, bg='#252526', highlightthickness=0)
        self.video_canvas.pack(pady=(0, 30))
        self.grid_canvas = tk.Canvas(left_frame, width=1200, height=450, bg='#000000', highlightthickness=0)
        self.grid_canvas.pack()

        right_frame = tk.Frame(main_frame, bg='#000000', width=600)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(50, 0))
        right_frame.pack_propagate(False)

        tk.Label(right_frame, text="Mode:", fg='#D4D4D4', bg='#000000', font=('Arial', 16, 'bold')).place(x=10, y=10)
        tk.Radiobutton(right_frame, text="Camera", variable=self.mode, value="camera", bg='#000000', fg='#D4D4D4', selectcolor='#252526', font=('Arial', 14), command=self._switch_mode).place(x=90, y=10)
        tk.Radiobutton(right_frame, text="Video", variable=self.mode, value="video", bg='#000000', fg='#D4D4D4', selectcolor='#252526', font=('Arial', 14), command=self._switch_mode).place(x=200, y=10)
        tk.Radiobutton(right_frame, text="Image", variable=self.mode, value="image", bg='#000000', fg='#D4D4D4', selectcolor='#252526', font=('Arial', 14), command=self._switch_mode).place(x=300, y=10)

        self.indicators = {
            'connected': StatusIndicator(right_frame, "Camera", 350, 60),
            'robot_detected': StatusIndicator(right_frame, "Bluetooth", 350, 140),
            'whiteboard_detected': StatusIndicator(right_frame, "Whiteboard", 350, 220)
        }

        tk.Label(right_frame, text="Camera ID:", fg='#D4D4D4', bg='#000000', font=('Arial', 14)).place(x=10, y=60)
        self.camera_id_entry = tk.Entry(right_frame, width=18, font=('Arial', 14), bg='#252526', fg='#D4D4D4')
        self.camera_id_entry.place(x=130, y=60)

        tk.Label(right_frame, text="Video Path:", fg='#D4D4D4', bg='#000000', font=('Arial', 14)).place(x=10, y=110)
        self.video_path_entry = tk.Entry(right_frame, width=18, font=('Arial', 14), bg='#252526', fg='#D4D4D4')
        self.video_path_entry.place(x=130, y=110)

        tk.Label(right_frame, text="Image Path:", fg='#D4D4D4', bg='#000000', font=('Arial', 14)).place(x=10, y=160)
        self.image_path_entry = tk.Entry(right_frame, width=18, font=('Arial', 14), bg='#252526', fg='#D4D4D4')
        self.image_path_entry.place(x=130, y=160)

        tk.Button(right_frame, text="Start Feed", bg='#616161', fg='#D4D4D4', font=('Arial', 14), width=12, command=self._start_feed).place(x=10, y=210)

        tk.Label(right_frame, text="BT Device:", fg='#D4D4D4', bg='#000000', font=('Arial', 14)).place(x=250, y=330)
        self.device_name_entry = tk.Entry(right_frame, width=18, font=('Arial', 14), bg='#252526', fg='#D4D4D4')
        self.device_name_entry.place(x=350, y=330)
        tk.Button(right_frame, text="Connect Robot", bg='#616161', fg='#D4D4D4', font=('Arial', 14), width=14, command=self._connect_robot).place(x=250, y=380)
        tk.Button(right_frame, text="Detect Board", bg='#616161', fg='#D4D4D4', font=('Arial', 14), width=14, command=self._detect_whiteboard).place(x=250, y=430)

        tk.Label(right_frame, text="Rows/Cols:", fg='#D4D4D4', bg='#000000', font=('Arial', 14)).place(x=10, y=280)
        self.rows_entry = tk.Entry(right_frame, width=6, font=('Arial', 14), bg='#252526', fg='#D4D4D4')
        self.rows_entry.place(x=130, y=280)
        self.cols_entry = tk.Entry(right_frame, width=6, font=('Arial', 14), bg='#252526', fg='#D4D4D4')
        self.cols_entry.place(x=200, y=280)
        tk.Button(right_frame, text="Update Grid", bg='#616161', fg='#D4D4D4', font=('Arial', 14), width=12, command=self._update_grid_size).place(x=10, y=330)

        tk.Label(right_frame, text="BLDC Speed:", fg='#D4D4D4', bg='#000000', font=('Arial', 14, 'bold')).place(x=250, y=600)
        self.bldc_vert_value_label = tk.Label(right_frame, text="1450", fg='#FFFFFF', bg='#000000', font=('Arial', 16, 'bold'))
        self.bldc_vert_value_label.place(x=270, y=630)
        self.bldc_vert_slider = tk.Scale(
            right_frame,
            from_=2000,
            to=1000,
            orient=tk.VERTICAL,
            length=280,
            sliderlength=30,
            width=30,
            troughcolor='#252526',
            bg='#000000',
            fg='#D4D4D4',
            highlightthickness=0,
            activebackground='#0D47A1',
            font=('Arial', 12),
            resolution=100,
            command=self._on_bldc_vert_slider_change
        )
        self.bldc_vert_slider.set(1000)
        self.bldc_vert_slider.place(x=270, y=660)

        tk.Label(right_frame, text="Mobile Upload:", fg='#D4D4D4', bg='#000000', font=('Arial', 14, 'bold')).place(x=10, y=390)
        if os.path.exists(QR_CODE_PATH):
            qr_img = Image.open(QR_CODE_PATH).resize((200, 200), Image.LANCZOS)
            self.qr_photo = ImageTk.PhotoImage(qr_img)
            tk.Label(right_frame, image=self.qr_photo, bg='#000000').place(x=10, y=430)
        
        tk.Button(right_frame, text="Sync Drive", bg='#4285F4', fg='#FFFFFF', font=('Arial', 12, 'bold'), width=12,
                  command=lambda: threading.Thread(target=sync_drive_to_local, daemon=True).start()).place(x=10, y=650)

        tk.Button(right_frame, text="START", bg='#0D47A1', fg='#FFFFFF', font=('Arial', 18, 'bold'), width=12,
                  command=self._on_start_clicked).place(x=250, y=510)
        
        tk.Button(right_frame, text="Erase Board", bg='#0D47A1', fg='#FFFFFF', font=('Arial', 18, 'bold'), width=12,
                  command=self._on_erase_board_clicked).place(x=250, y=570)

        emergency_stop_canvas = tk.Canvas(right_frame, width=150, height=150, bg='#000000', highlightthickness=0)
        emergency_stop_canvas.place(x=420, y=460)
        
        emergency_stop_canvas.create_oval(10, 10, 130, 130, fill='#D32F2F', outline='#E53935', width=5)
        emergency_stop_canvas.create_text(70, 45, text="EMERGENCY", fill='#FFFFFF', font=('Arial', 11, 'bold'))
        emergency_stop_canvas.create_text(70, 85, text="STOP", fill='#FFFFFF', font=('Arial', 20, 'bold'))
        
        emergency_stop_canvas.tag_bind('all', '<Button-1>', lambda e: self._on_emergency_stop_clicked())
        emergency_stop_canvas.config(cursor="hand2")

        self.dpad = DirectionalPad(right_frame, 10, 690, self.robot_controller)

        listbox_frame = tk.Frame(right_frame, bg='#000000')
        listbox_frame.place(x=370, y=650, width=180, height=180)

        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_listbox = tk.Listbox(
            listbox_frame,
            bg='#252526',
            fg='#D4D4D4',
            font=('Arial', 12),
            selectbackground='#0D47A1',
            yscrollcommand=scrollbar.set
        )
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.image_listbox.yview)

        self.image_listbox.bind('<<ListboxSelect>>', self._on_image_selected)

        tk.Button(right_frame, text="Refresh Images", bg='#4285F4', fg='#FFFFFF', font=('Arial', 10, 'bold'),
                  command=self._load_whiteboard_images).place(x=10, y=1000)

    def _check_confirm(self):
        if self.robot_controller.need_confirm:
            self.robot_controller.need_confirm = False
            result = messagebox.askyesno("Confirmation", self.robot_controller.confirm_message)
            self.robot_controller.confirm_result = result
            self.robot_controller.confirm_event.set()
        self.root.after(100, self._check_confirm)

    def _on_emergency_stop_clicked(self):
        print("=== EMERGENCY STOP ACTIVATED! ===")
        
        if self.robot_controller.ser and self.robot_controller.ser.is_open:
            send_command(self.robot_controller.ser, "1000\n")
            print("Sent BLDC speed: 1000 (STOPPED)")
            send_command(self.robot_controller.ser, 'S')
            print("Sent DC motors STOP command: 'S'")
        else:
            print("Bluetooth not connected! Cannot send emergency stop.")

        self.robot_controller.stop_cleaning()
        
        self.bldc_vert_slider.set(1000)
        self.bldc_vert_value_label.config(text="1000")

    def _load_whiteboard_images(self):
        folder = LOCAL_IMAGE_FOLDER
        if not os.path.exists(folder):
            return
        self.image_listbox.delete(0, tk.END)
        self.whiteboard_image_paths = []
        for file in sorted(os.listdir(folder)):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                full_path = os.path.join(folder, file)
                self.whiteboard_image_paths.append(full_path)
                self.image_listbox.insert(tk.END, file)
        if self.whiteboard_image_paths:
            self.image_listbox.select_set(0)
            self._on_image_selected(None)

    def _on_image_selected(self, event):
        selection = self.image_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        image_path = self.whiteboard_image_paths[idx]
        print(f"Selected image: {image_path}")

        if self.camera:
            self.camera.stop()
            self.camera = None

        temp_camera = CameraFeed(source='image', image_path=image_path)
        if temp_camera.start():
            if self.camera is None:
                self.camera = temp_camera
            else:
                with temp_camera.lock:
                    self.camera.current_frame = temp_camera.current_frame
                    self.camera.grid_image = temp_camera.grid_image
                    self.camera.whiteboard_detected = temp_camera.whiteboard_detected
                    self.camera.whiteboard_bbox = temp_camera.whiteboard_bbox
                    self.camera.whiteboard_h = temp_camera.whiteboard_h
                    self.camera.whiteboard_w = temp_camera.whiteboard_w
                    self.camera.dirty_cells = temp_camera.dirty_cells.copy()
                    self.camera.robot_pose = temp_camera.robot_pose

            self.status_manager.set_status('connected', True)
            self.status_manager.set_status('whiteboard_detected', self.camera.whiteboard_detected)

            self._update_camera_feed()
            self._update_grid_feed()

            self._run_handwritten_detection(image_path)

    def _run_handwritten_detection(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            print("Failed to load image for detection")
            return

        frame = cv2.resize(frame, (1200, 450))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.camera.model.predict(frame, device='cpu', verbose=False, conf=0.01)

        whiteboard_found = False
        whiteboard_bbox = None
        self.camera.dirty_cells.clear()

        for r in results:
            boxes = r.boxes
            for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                class_name = self.camera.model.names[int(cls)].lower()
                print(f"Detected: {class_name} with confidence {conf}")

                if conf < 0.01:
                    continue

                x1, y1, x2, y2 = map(int, box)

                if 'whiteboard' in class_name:
                    whiteboard_found = True
                    margin = 20
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    whiteboard_bbox = (y1, y2, x1, x2)
                    whiteboard_frame = frame[y1:y2, x1:x2]
                    self.camera.whiteboard_h, self.camera.whiteboard_w = whiteboard_frame.shape[:2]

                elif 'handwritten' in class_name or 'text' in class_name:
                    if whiteboard_found and whiteboard_bbox:
                        wy1, wy2, wx1, wx2 = whiteboard_bbox
                        rel_x1 = x1 - wx1
                        rel_y1 = y1 - wy1
                        rel_x2 = x2 - wx1
                        rel_y2 = y2 - wy1

                        if rel_x1 < 0 or rel_y1 < 0 or rel_x2 > self.camera.whiteboard_w or rel_y2 > self.camera.whiteboard_h:
                            continue

                        cell_h = self.camera.whiteboard_h / self.camera.num_rows
                        cell_w = self.camera.whiteboard_w / self.camera.num_cols

                        start_row = int(rel_y1 / cell_h)
                        end_row = int(rel_y2 / cell_h) + 1
                        start_col = int(rel_x1 / cell_w)
                        end_col = int(rel_x2 / cell_w) + 1

                        for row in range(max(0, start_row), min(self.camera.num_rows, end_row)):
                            for col in range(max(0, start_col), min(self.camera.num_cols, end_col)):
                                if self.camera.border <= row < self.camera.num_rows - self.camera.border and self.camera.border <= col < self.camera.num_cols - self.camera.border:
                                    self.camera.dirty_cells.add((row, col))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        if whiteboard_found:
            self.camera.whiteboard_bbox = whiteboard_bbox
            annotated = cv2.resize(frame[whiteboard_bbox[0]:whiteboard_bbox[1], whiteboard_bbox[2]:whiteboard_bbox[3]], (1200, 500))
            self.camera.current_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            self.camera.whiteboard_detected = True
            self.status_manager.set_status('whiteboard_detected', True)
        else:
            self.camera.current_frame = rgb_frame
            self.camera.whiteboard_detected = False
            self.status_manager.set_status('whiteboard_detected', False)

        self.camera.compute_grid_image()
        self._update_camera_feed()
        self._update_grid_feed()
        self.saved_dirty_cells = self.camera.dirty_cells.copy()

    def cleanup(self):
        if self.camera:
            self.camera.stop()
        self.robot_controller.cleanup()

    def _start_feed(self):
        if self.camera:
            self.camera.stop()
        mode = self.mode.get()
        if mode == 'camera':
            try:
                camera_id = int(self.camera_id_entry.get())
            except ValueError:
                print("Invalid camera ID")
                return
            self.camera = CameraFeed(source=mode, camera_id=camera_id)
        elif mode == 'video':
            video_path = self.video_path_entry.get()
            if not os.path.exists(video_path):
                print("Video file not found")
                return
            self.camera = CameraFeed(source=mode, video_path=video_path)
        elif mode == 'image':
            image_path = self.image_path_entry.get()
            if not os.path.exists(image_path):
                print("Image file not found")
                return
            self.camera = CameraFeed(source=mode, image_path=image_path)
        if self.camera.start():
            if mode in ['camera', 'video']:
                self.camera.dirty_cells = self.saved_dirty_cells.copy()
            self.status_manager.set_status('connected', True)
            self._update_camera_feed()
            self._update_grid_feed()
        else:
            self.status_manager.set_status('connected', False)
            self._show_placeholder()
    
    def _switch_mode(self):
        self._start_feed()
    
    def _connect_robot(self):
        device_name = self.device_name_entry.get()
        if self.robot_controller.connect(device_name):
            self.status_manager.set_status('robot_detected', True)
    
    def _detect_whiteboard(self):
        if self.camera is None or self.camera.current_frame is None:
            print("No feed available")
            return
        frame = cv2.cvtColor(self.camera.get_frame(), cv2.COLOR_RGB2BGR)
        results = self.camera.model.predict(frame, device='cpu', verbose=False, conf=0.01)
        for r in results:
            boxes = r.boxes
            for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                class_name = self.camera.model.names[int(cls)].lower()
                print(f"Live detect: {class_name} conf {conf}")
                if conf < 0.01:
                    continue
                if 'whiteboard' in class_name:
                    x1, y1, x2, y2 = map(int, box)
                    margin = 20
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    self.camera.whiteboard_bbox = (y1, y2, x1, x2)
                    whiteboard_frame = frame[y1:y2, x1:x2]
                    self.camera.whiteboard_h, self.camera.whiteboard_w = whiteboard_frame.shape[:2]
                    self.camera.compute_grid_image()
                    self.status_manager.set_status('whiteboard_detected', True)
                    self._plan_path()
                    return
    
    def _update_grid_size(self):
        try:
            rows = int(self.rows_entry.get())
            cols = int(self.cols_entry.get())
            if rows > 0 and cols > 0:
                self.grid_manager.update_grid_size(rows, cols)
                if self.camera:
                    self.camera.num_rows = rows
                    self.camera.num_cols = cols
                    self.camera.compute_grid_image()
        except ValueError:
            print("Invalid grid size input")
    
    def _update_camera_feed(self):
        frame = self.camera.get_frame() if self.camera else None
        if frame is not None:
            img = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=img)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.video_canvas.image = photo
        if self.camera and self.camera.running:
            self.root.after(30, self._update_camera_feed)
    
    def _update_grid_feed(self):
        grid_image = self.camera.get_grid_image() if self.camera else None
        if grid_image is not None:
            img = Image.fromarray(grid_image)
            photo = ImageTk.PhotoImage(image=img)
            self.grid_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.grid_canvas.image = photo
        if self.camera and self.camera.running:
            self.root.after(30, self._update_grid_feed)
    
    def _show_placeholder(self):
        img = Image.new('RGB', (1200, 450), color='#252526')
        photo = ImageTk.PhotoImage(img)
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.video_canvas.image = photo
        self.video_canvas.create_text(600, 250, text="Camera/Video/Image Not Available",
                                     fill='#D4D4D4', font=('Arial', 28))
    
    def _on_status_update(self, statuses):
        self.indicators['connected'].set_active(statuses['connected'])
        self.indicators['robot_detected'].set_active(statuses['robot_detected'])
        self.indicators['whiteboard_detected'].set_active(statuses['whiteboard_detected'])
    
    def _on_start_clicked(self):
        if not self.status_manager.get_status('connected'):
            print("System not ready for cleaning")
            return
        self.robot_controller.start_cleaning()
        threading.Thread(target=self.robot_controller.auto_clean,
                         args=(self.camera.dirty_cells, self.camera.robot_pose, self.camera.num_rows, self.camera.num_cols, self.camera),
                         daemon=True).start()
        print("Start button clicked")
    
    def _on_erase_board_clicked(self):
        if not (self.status_manager.get_status('connected') and
                self.status_manager.get_status('robot_detected') and
                self.status_manager.get_status('whiteboard_detected') and
                self.camera and self.camera.robot_pose):
            print("System not ready for cleaning")
            return
        if self.camera:
            border = self.camera.border
            all_cells = set((r, c) for r in range(border, self.camera.num_rows - border) for c in range(border, self.camera.num_cols - border))
            self.camera.dirty_cells = all_cells
        if not self.camera.planned_path:
            self._plan_path()
        self.robot_controller.start_cleaning()
        threading.Thread(target=self.robot_controller.whiteboard_clean,
                         args=(self.camera.robot_pose, self.camera.num_rows, self.camera.num_cols, self.camera),
                         daemon=True).start()
        print("Erase Board started")
    
    def _on_bldc_vert_slider_change(self, value):
        val = int(float(value))
        self.bldc_vert_value_label.config(text=str(val))
        if self.robot_controller.ser and self.robot_controller.ser.is_open:
            command = f"{val}\n"
            success = send_command(self.robot_controller.ser, command)
            if success:
                print(f"Sent BLDC speed: {val}")
            else:
                print("Failed to send BLDC speed")
        else:
            print("Bluetooth not connected! Cannot send BLDC speed.")

    def _plan_path(self):
        if self.camera:
            order = self.robot_controller.get_coverage_order(self.camera.num_rows, self.camera.num_cols, self.camera.border)
            self.camera.planned_path = order
            self.camera.compute_grid_image()
            self._update_grid_feed()
            
            
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1920x1080")
    root.configure(bg='#000000')
    app = RobotControlGUI(root)
    
    def on_closing():
        app.cleanup()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


    

