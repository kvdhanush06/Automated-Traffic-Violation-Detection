import cv2
import time
from datetime import datetime
import os
import csv
import numpy as np
from ultralytics import YOLO
import torch.serialization
import ultralytics.nn.tasks as nn_tasks
from sort import Sort
from ocr_utils import extract_number_plate_text

VIDEO_SOURCE = 0
LINE_A_Y = 200
LINE_B_Y = 400
DISTANCE_METERS = 5.0
SPEED_LIMIT_KMPH = 30
FRAME_SKIP = 1
MAX_TRACKED_VEHICLES = 50
MIN_VIDEO_DURATION = 10.0

CAPTURE_DIR = 'static/captures'
LOG_FILE = 'log.csv'

os.makedirs(CAPTURE_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Time', 'Speed_limit_kmph',
                        'Speed_kmph', 'Plate_text', 'Plate_image'])

try:
    torch.serialization.add_safe_globals([nn_tasks.DetectionModel])
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')
    coco_model.verbose = False
    license_plate_detector.verbose = False
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

vehicles = [2, 3, 5, 7]
vehicle_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
mot_tracker = Sort()


def try_open_camera(preferred_source):
    sources_to_try = [preferred_source]

    # Add other common indices
    if isinstance(preferred_source, int):
        for i in range(5):
            if i not in sources_to_try:
                sources_to_try.append(i)
    elif isinstance(preferred_source, str) and preferred_source.isdigit():
        pref_int = int(preferred_source)
        sources_to_try = [pref_int]
        for i in range(5):
            if i != pref_int:
                sources_to_try.append(i)
    else:
        # If it's a file, just try once
        cap = cv2.VideoCapture(preferred_source)
        return cap, preferred_source

    for src in sources_to_try:
        print(f"Trying camera source: {src}")
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Successfully opened camera {src}")
                return cap, src
            else:
                cap.release()
        else:
            print(f"Failed to open camera {src}")
    return None, None


def run_detector(video_source=VIDEO_SOURCE, headless=False, speed_limit=SPEED_LIMIT_KMPH,
                 only_overspeed=False, debug=True):

    # Determine if we're using a camera or video file
    is_camera = isinstance(video_source, int) or (
        isinstance(video_source, str) and video_source.isdigit())

    if is_camera:
        cap, actual_source = try_open_camera(
            int(video_source) if isinstance(video_source, str) else video_source)
        if cap is None:
            print('ERROR: Unable to open any camera source')
            return
        print(f"Using camera source: {actual_source}")
    else:
        print(f"Using video file: {video_source}")
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f'ERROR: Unable to open video source: {video_source}')
            return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if is_camera:
        total_frames = -1  # Cameras don't have a fixed frame count
        print(f"Camera opened: {width}x{height} @ {fps} FPS")
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(
            f"Video file opened: {width}x{height} @ {fps} FPS, {total_frames} total frames")

    tracked = {}
    max_disappeared_seconds = 2.0
    frame_count = 0
    video_restarts = 0
    last_cleanup_time = time.time()
    video_start_time = time.time()
    consecutive_read_failures = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            consecutive_read_failures += 1

            if is_camera:
                # For cameras, handle read failures more gracefully
                if consecutive_read_failures >= 10:  # Increased threshold for cameras
                    current_video_duration = time.time() - video_start_time
                    print(
                        f"Camera read failure (attempt {consecutive_read_failures}, duration: {current_video_duration:.1f}s)")
                    time.sleep(0.5)  # Longer delay for camera recovery
                    continue
                else:
                    time.sleep(0.1)
                    continue
            else:
                # Original logic for video files
                if consecutive_read_failures >= 3:
                    current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    current_video_duration = time.time() - video_start_time

                    if (current_video_duration >= MIN_VIDEO_DURATION and
                            (current_frame_pos >= total_frames - 5 or current_frame_pos < 0)):
                        print(
                            f"Video ended (frame {current_frame_pos}/{total_frames}, duration: {current_video_duration:.1f}s), restarting...")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        video_restarts += 1
                        video_start_time = time.time()
                        consecutive_read_failures = 0

                        ret, frame = cap.read()
                        if not ret:
                            print(
                                "ERROR: Cannot restart video after setting position to 0")
                            break
                    else:
                        if debug and consecutive_read_failures % 10 == 0:
                            print(
                                f"Waiting for video end... (failures: {consecutive_read_failures}, duration: {current_video_duration:.1f}s)")
                        time.sleep(0.1)
                        continue
                else:
                    continue
        else:
            consecutive_read_failures = 0

        current_time = time.time()
        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        try:
            detections = coco_model(frame)[0]
            detections_ = []

            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles and score > 0.3:
                    detections_.append([x1, y1, x2, y2, score])

            track_ids = mot_tracker.update(np.asarray(detections_))

            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(
                    license_plate, track_ids)

                if car_id != -1:
                    try:
                        license_plate_crop = frame[int(
                            y1):int(y2), int(x1):int(x2), :]
                        license_plate_crop_gray = cv2.cvtColor(
                            license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(
                            license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                        license_plate_text, _ = extract_number_plate_text(
                            license_plate_crop_thresh)

                        if license_plate_text and license_plate_text.strip():
                            if car_id not in tracked:
                                tracked[car_id] = {
                                    'start_time': None, 'start_image': None,
                                    'plate_text': license_plate_text, 'recorded': False,
                                    'last_seen': current_time, 'seen': True
                                }
                            else:
                                tracked[car_id]['plate_text'] = license_plate_text
                                tracked[car_id]['last_seen'] = current_time
                                tracked[car_id]['seen'] = True
                    except Exception as e:
                        if debug:
                            print(f"Error processing license plate: {e}")

            for track in track_ids:
                x1, y1, x2, y2, car_id = track
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                if car_id not in tracked:
                    tracked[car_id] = {
                        'centroid': (cx, cy), 'last_y': cy, 'start_time': None,
                        'start_image': None, 'plate_text': None, 'recorded': False,
                        'last_seen': current_time, 'seen': True
                    }
                else:
                    prev_y = tracked[car_id].get('last_y', cy)
                    tracked[car_id]['centroid'] = (cx, cy)
                    tracked[car_id]['last_y'] = cy
                    tracked[car_id]['last_seen'] = current_time
                    tracked[car_id]['seen'] = True

                    if prev_y < LINE_A_Y <= cy and tracked[car_id]['start_time'] is None:
                        tracked[car_id]['start_time'] = current_time
                        tracked[car_id]['start_image'] = frame.copy()

                    if (prev_y < LINE_B_Y <= cy and tracked[car_id]['start_time'] is not None
                            and not tracked[car_id]['recorded']):

                        t1 = tracked[car_id]['start_time']
                        elapsed = max(0.1, current_time - t1)

                        if elapsed > 0.1:
                            speed_mps = DISTANCE_METERS / elapsed
                            speed_kmph = speed_mps * 3.6

                            if 1 <= speed_kmph <= 200:
                                timestamp_dt = datetime.now()
                                date_str = timestamp_dt.strftime('%Y-%m-%d')
                                time_str = timestamp_dt.strftime('%H:%M:%S')
                                img_filename = f"{timestamp_dt.strftime('%Y%m%d_%H%M%S')}_{car_id}.jpg"
                                img_path = os.path.join(
                                    CAPTURE_DIR, img_filename)

                                try:
                                    cv2.imwrite(
                                        img_path, tracked[car_id]['start_image'])
                                except Exception as e:
                                    print(f"Error saving image: {e}")
                                    continue

                                plate_text = tracked[car_id].get(
                                    'plate_text', '')

                                if speed_kmph > speed_limit or not only_overspeed:
                                    try:
                                        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                                            writer = csv.writer(f)
                                            writer.writerow([
                                                date_str, time_str, f"{speed_limit:.2f}",
                                                f"{speed_kmph:.2f}", plate_text, img_path
                                            ])
                                    except Exception as e:
                                        print(f"Error writing to CSV: {e}")

                                print(
                                    f"[CAPTURE] id={car_id} speed={speed_kmph:.2f} km/h plate={plate_text}")
                                tracked[car_id]['recorded'] = True

            if current_time - last_cleanup_time > 10:
                to_del = []
                for tid in tracked:
                    if not tracked[tid].get('seen', False):
                        if current_time - tracked[tid].get('last_seen', 0) > max_disappeared_seconds:
                            to_del.append(tid)

                for tid in to_del:
                    del tracked[tid]

                if len(tracked) > MAX_TRACKED_VEHICLES:
                    sorted_tracked = sorted(
                        tracked.items(), key=lambda x: x[1]['last_seen'])
                    to_remove = sorted_tracked[:len(
                        tracked) - MAX_TRACKED_VEHICLES]
                    for tid, _ in to_remove:
                        del tracked[tid]

                last_cleanup_time = current_time

            for tid in tracked:
                tracked[tid]['seen'] = False

            cv2.line(frame, (0, LINE_A_Y),
                     (frame.shape[1], LINE_A_Y), (0, 255, 0), 2)
            cv2.line(frame, (0, LINE_B_Y),
                     (frame.shape[1], LINE_B_Y), (0, 0, 255), 2)

            for track in track_ids:
                x1, y1, x2, y2, car_id = track
                cv2.rectangle(frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"ID:{int(car_id)}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            try:
                latest_path = os.path.join('static', 'latest.jpg')
                cv2.imwrite(latest_path, frame)
            except Exception as e:
                if debug:
                    print(f"Error saving latest frame: {e}")

            if not headless:
                cv2.imshow("Speed Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            continue

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    print(f"Detector stopped. Processed {frame_count} frames.")


def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    for track in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = track
        if (x1 > xcar1 - 10 and y1 > ycar1 - 10 and
                x2 < xcar2 + 10 and y2 < ycar2 + 10):
            return xcar1, ycar1, xcar2, ycar2, car_id

    return -1, -1, -1, -1, -1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Speed detector and camera test helper')
    parser.add_argument('--test-camera', action='store_true',
                        help='Scan common camera indices and try to open a camera (diagnostic)')
    parser.add_argument('--headless', action='store_true',
                        help='Run detector in headless mode (no GUI)')
    parser.add_argument('--video-source', type=str,
                        default=str(VIDEO_SOURCE), help='Video source (index or path)')
    parser.add_argument('--speed-limit', type=float, default=float(SPEED_LIMIT_KMPH),
                        help='Speed limit (km/h) used for calculation and logging')
    parser.add_argument('--only-overspeed', action='store_true',
                        help='Only write to CSV when measured speed exceeds the speed limit')
    parser.add_argument('--append-dummy', action='store_true',
                        help='Create a dummy capture image and CSV row for testing the dashboard')
    parser.add_argument('--debug', action='store_true', default=True,
                        help='Enable debug output')
    args = parser.parse_args()

    if args.test_camera:
        print('Running camera diagnostic...')

        def quick_test(src):
            cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f'Camera {src} working: {frame.shape}')
                else:
                    print(f'Camera {src} opened but no frame received')
                cap.release()
            else:
                print(f'Camera {src} failed to open')

        for i in range(5):
            quick_test(str(i))

    elif args.append_dummy:
        timestamp_dt = datetime.now()
        date_str = timestamp_dt.strftime('%Y-%m-%d')
        time_str = timestamp_dt.strftime('%H:%M:%S')
        img_filename = f"dummy_{timestamp_dt.strftime('%Y%m%d_%H%M%S')}.jpg"
        img_path = os.path.join(CAPTURE_DIR, img_filename)

        h, w = 360, 640
        img = np.full((h, w, 3), 200, dtype=np.uint8)
        cv2.putText(img, 'DUMMY PLATE: TEST123', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(img, f'SPEED: 55.0 km/h', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        cv2.imwrite(img_path, img)

        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(
                [date_str, time_str, f"{args.speed_limit:.2f}", "55.0", 'TEST123', img_path])

        print('Dummy data added to log.csv and captures/')

    else:
        run_detector(
            video_source=int(
                args.video_source) if args.video_source.isdigit() else args.video_source,
            headless=args.headless,
            speed_limit=args.speed_limit,
            only_overspeed=args.only_overspeed,
            debug=args.debug
        )
