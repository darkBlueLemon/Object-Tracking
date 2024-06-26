from tabnanny import verbose
from zmq import device
from ultralytics import YOLO
import cv2
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
from concurrent.futures import ThreadPoolExecutor
import time
import os
import random

results = {}

mot_tracker = Sort()

# Load models with batch processing optimization
coco_model = YOLO('./models/yolov8x.pt')
coco_model.to(device="mps")
license_plate_detector = YOLO('./models/bestNew.pt')
license_plate_detector.to(device="mps")

# Load video
# cap = cv2.VideoCapture('./videos/sample10.mp4')
cap = cv2.VideoCapture(1)

# vehicles = [2, 3, 5, 7]
vehicles = [0]

# Define the horizontal segment
y_start, y_end = 0.3, 0.9
_, frame = cap.read()
y_start, y_end = int(frame.shape[0] * y_start), int(frame.shape[0] * y_end)
x_start, x_end = 0, frame.shape[1]
frame_width = frame.shape[1]

y_line = int(frame.shape[1] * 0.5)  # Adjust this value to set the line position

# Store previous positions and direction counts of tracked vehicles
prev_positions = {}
direction_counts = {}
threshold_frames = 10  # Number of frames a vehicle must move in one direction
track_id_plate = {}
track_id_entry_exit = {}
track_id_wrong_side = {}

# Left Hand Side Driving
left_driving = False

# Read frames
frame_nmr = -1
ret = True

def process_frame(frame, frame_nmr, start_time):
    results[frame_nmr] = {}

    # Crop the frame to the area of interest
    cropped_frame = frame[y_start:y_end, x_start:x_end]

    # Detect vehicles in the cropped frame
    detections = coco_model(cropped_frame, verbose=False)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            # Adjust coordinates to match the original frame
            x1 += x_start
            x2 += x_start
            y1 += y_start
            y2 += y_start
            detections_.append([x1, y1, x2, y2, score])

    # Convert detections_ to a NumPy array
    detections_ = np.asarray(detections_)

    # Check if detections_ is empty before updating the tracker
    if detections_.shape[0] > 0:
        # Track vehicles
        track_ids = mot_tracker.update(detections_)

        # Draw vehicle boxes
        for track in track_ids:
            x1, y1, x2, y2, track_id = track
            if not np.isnan([x1, y1, x2, y2, track_id]).any():
                # Calculate the center of the bounding box
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

                # Determine the movement direction
                if track_id in prev_positions:
                    prev_center_x, prev_center_y = prev_positions[track_id]
                    if center_y < prev_center_y:
                        direction = 'EXIT'
                    else:
                        direction = 'ENTRY'

                    # Update direction counts
                    if track_id in direction_counts:
                        last_direction, count = direction_counts[track_id]
                        if direction == last_direction:
                            direction_counts[track_id] = (last_direction, count + 1)
                        else:
                            direction_counts[track_id] = (direction, 1)
                    else:
                        direction_counts[track_id] = (direction, 1)
                else:
                    direction = None
                    direction_counts[track_id] = (None, 0)

                # Update previous positions
                prev_positions[track_id] = (center_x, center_y)

                # Determine color based on direction and count
                color = (255, 255, 255)  # White for the first frame
                if direction is not None:
                    last_direction, count = direction_counts[track_id]
                    if count >= threshold_frames:
                        if last_direction == 'ENTRY':
                            color = (0, 255, 0)  # Green for moving away
                            cv2.putText(frame, f'ID: {int(track_id)} ENTRY', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        elif last_direction == 'EXIT':
                            color = (0, 0, 255)  # Red for coming towards
                            cv2.putText(frame, f'ID: {int(track_id)} EXIT', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                        # Adding car ID with entry/exit to car_id_entry_exit
                        if track_id in track_id_plate:
                            track_id_entry_exit[track_id] = {track_id_plate[track_id], last_direction}

                        # # Comment for normal use
                        # color = (255, 255, 255)
                        # if left_driving:
                        #     cv2.putText(frame, 'ENTRY', (int(y_line), int(y_end) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        #     cv2.putText(frame, 'EXIT', (int(0), int(y_end) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        #     if (center_x < y_line and last_direction == 'ENTRY') or (center_x > y_line and last_direction == 'EXIT'):
                        #         color = (0, 0, 255)
                        #         # Adding car ID with wrong side
                        #         if track_id in track_id_plate:
                        #             track_id_wrong_side[track_id] = {track_id_plate[track_id], last_direction}
                        # else:
                        #     cv2.putText(frame, 'EXIT', (int(y_line), int(y_end) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        #     cv2.putText(frame, 'ENTRY', (int(0), int(y_end) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        #     if (center_x < y_line and last_direction == 'EXIT') or (center_x > y_line and last_direction == 'ENTRY'):
                        #         color = (0, 0, 255)
                        #         # Adding car ID with wrong side
                        #         if track_id in track_id_plate:
                        #             track_id_wrong_side[track_id] = {track_id_plate[track_id], last_direction}

                # Draw the bounding box and center point
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)

        # Detect license plates
        license_plates = license_plate_detector(frame, verbose=False)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                if license_plate_crop.size == 0:
                    continue

                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 10)
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

                    track_id_plate[car_id] = license_plate_text

                    # Draw license plate box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw bounding lines
    cv2.line(frame, (x_start, y_start), (x_end, y_start), (0, 255, 0), 2)
    cv2.line(frame, (x_start, y_end), (x_end, y_end), (0, 255, 0), 2)
    cv2.line(frame, (y_line, y_start), (y_line, y_end), (0, 255, 0), 2)

    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

with ThreadPoolExecutor() as executor:
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            start_time = time.time()
            future = executor.submit(process_frame, frame, frame_nmr, start_time)
            frame = future.result()
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

# Write results
print(track_id_plate)
print(track_id_entry_exit)
write_csv(results, './test.csv')

cap.release()
cv2.destroyAllWindows()
