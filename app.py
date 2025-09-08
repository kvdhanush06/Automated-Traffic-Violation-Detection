from flask import Flask, render_template, Response
import csv
import os
import cv2
import numpy as np
import threading
from speed_detector import run_detector

app = Flask(__name__,
            static_folder='static',
            static_url_path='/static')

LOG_FILE = 'log.csv'

latest_frame = None
frame_lock = threading.Lock()


def generate_frames():
    import time
    import os

    while True:
        try:
            frame_path = os.path.join(os.getcwd(), 'static', 'latest.jpg')

            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                frame = cv2.imread(frame_path)
                if frame is not None and frame.size > 0:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        pass
                else:
                    pass
            else:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for video feed...", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error in generate_frames: {e}")
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Video feed error", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            try:
                ret, buffer = cv2.imencode('.jpg', error_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except:
                pass

        time.sleep(0.05)


@app.route('/')
def index():
    return 'Flask App Running. Visit /dashboard to view overspeeding vehicles.'


@app.route('/dashboard')
def dashboard():
    entries = []

    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                print(f"CSV headers: {reader.fieldnames}")
                for row in reader:
                    print(f"Raw row: {row}")
                    processed_row = {
                        'date': row.get('Date', '').strip(),
                        'time': row.get('Time', '').strip(),
                        'speed_limit': float(row.get('Speed_limit_kmph', '0').strip() or 0),
                        'speed_kmph': float(row.get('Speed_kmph', '0').strip() or 0),
                        'plate_text': row.get('Plate_text', '').strip(),
                        'plate_image': row.get('Plate_image', '').strip()
                    }
                    entries.append(processed_row)
                    print(f"Processed row: {processed_row}")
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return f"Error reading log file: {e}"

    print(f"Total entries: {len(entries)}")

    entries.reverse()

    return render_template('dashboard.html', violations=entries)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)


def run_detector_thread():
    global latest_frame

    def frame_callback(frame):
        with frame_lock:
            latest_frame = frame.copy()

    run_detector(headless=True)


def run_app(host='0.0.0.0', port=5000, debug=False):
    detector_thread = threading.Thread(target=run_detector_thread, daemon=True)
    detector_thread.start()

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_app(debug=True)
