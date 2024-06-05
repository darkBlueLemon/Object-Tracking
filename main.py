from tabnanny import verbose
from zmq import device
from ultralytics import YOLO
import cv2
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
coco_model.to(device="mps")
license_plate_detector = YOLO('./models/bestNew.pt')
license_plate_detector.to(device="mps")

# load video
cap = cv2.VideoCapture('./sample5r.mp4')
# cap = cv2.VideoCapture(0)

vehicles = [2, 3, 5, 7]
# Car
# vehicles = [2]

# Define the horizontal segment
y_start, y_end = 0.6, 0.8
_, frame = cap.read()
y_start, y_end = int(frame.shape[0]*y_start), int(frame.shape[0]*y_end)
x_start, x_end = 0, frame.shape[1]

# Store previous positions and direction counts of tracked vehicles
prev_positions = {}
direction_counts = {}
threshold_frames = 5  # Number of frames a vehicle must move in one direction
track_id_plate = {}
track_id_entry_exit = {}

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame, verbose=False)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            # if int(class_id) in vehicles:
                # detections_.append([x1, y1, x2, y2, score])
            # if int(class_id) in vehicles and y_start <= y1 <= y_end and y_start <= y2 <= y_end:
            #     detections_.append([x1, y1, x2, y2, score])
            if (int(class_id) in vehicles) and (y_start <= (y2+y1)/2 <= y_end):
                detections_.append([x1, y1, x2, y2, score])

        # convert detections_ to a NumPy array
        detections_ = np.asarray(detections_)

        # check if detections_ is empty before updating the tracker
        if detections_.shape[0] > 0:
            # Filter out detections with NaN values
            # detections_ = detections_[~np.isnan(detections_).any(axis=1)]
            # track vehicles
            track_ids = mot_tracker.update(detections_)

            # draw vehicle boxes
            # draw vehicle boxes
            for track in track_ids:
                x1, y1, x2, y2, track_id = track
                if not np.isnan([x1, y1, x2, y2, track_id]).any():
                    # Calculate the center of the bounding box
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

                    # Determine the movement direction
                    if track_id in prev_positions:
                        prev_center_x, prev_center_y = prev_positions[track_id]
                        if center_y < prev_center_y:
                            direction = 'up'
                        else:
                            direction = 'down'

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

                    # Determine color based on direction and count
                    color = (255, 255, 255)  # White for the first frame
                    if direction is not None:
                        last_direction, count = direction_counts[track_id]
                        if count >= threshold_frames:
                            if last_direction == 'down':
                                color = (0, 255, 0)  # Green for moving away
                            elif last_direction == 'up':
                                color = (0, 0, 255)  # Red for coming towards

                    # Update previous positions
                    prev_positions[track_id] = (center_x, center_y)

                    # Draw the bounding box and center point
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)
                    # cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    if color == (0, 255, 0):
                        action = "ENTRY"
                        cv2.putText(frame, f'ID: {int(track_id)} ENTRY', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    elif color == (0, 0, 255):
                        action = "EXIT"
                        cv2.putText(frame, f'ID: {int(track_id)} EXIT', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.circle(frame, (int(center_x), int(center_y)), radius=10, color=color, thickness=2)

                    # adding car id with entry/exit to car_id_extry_exit
                    if track_id in track_id_plate and action:
                        track_id_entry_exit[track_id] = {track_id_plate[track_id], action}
            # # draw vehicle boxes
            # for track in track_ids:
            #     x1, y1, x2, y2, track_id = track
            #     if not np.isnan([x1, y1, x2, y2, track_id]).any():
            #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
            #         cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
            #         cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), radius=10, color=(0, 0, 255), thickness=50)
            
            # detect license plates
            license_plates = license_plate_detector(frame, verbose=False)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                # track_id = get_car(license_plate, track_ids)
                # xcar1, ycar1, xcar2, ycar2, car_id = track_ids[track_id]

                if car_id != -1:

                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                    if license_plate_crop.size == 0:
                        continue

                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 10)
                    # cv2.putText(frame, license_plate_text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 10)
                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score},
                        }

                        track_id_plate[car_id] = license_plate_text
                        
                        # draw license plate box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, license_plate_text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                

        # draw bouding lines
        cv2.line(frame, (x_start, y_start), (x_end, y_start), (0, 255, 0), 2)
        cv2.line(frame, (x_start, y_end), (x_end, y_end), (0, 255, 0), 2)

        # display the frame with boxes
        cv2.imshow('Frame', frame)
        
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# write results
print("AAAAA")
print(track_id_plate)
print(track_id_entry_exit)
write_csv(results, './test.csv')