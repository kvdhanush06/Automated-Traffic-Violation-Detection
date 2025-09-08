import multiprocessing
import signal
import time
import argparse
import sys

from app import run_app
from speed_detector import run_detector


def _run_flask(host, port):
    run_app(host=host, port=port, debug=False)


def _run_detector(video_source, headless, speed_limit):
    if isinstance(video_source, str) and video_source.isdigit():
        vs = int(video_source)
    else:
        vs = video_source

    run_detector(video_source=vs,
                 headless=headless,
                 speed_limit=speed_limit)


def main():
    parser = argparse.ArgumentParser(
        description='Run Flask dashboard and speed detector')
    parser.add_argument('--host', default='0.0.0.0', help='Flask host')
    parser.add_argument('--port', type=int, default=5000, help='Flask port')
    parser.add_argument('--video', default='0',
                        help='Video source: path to video file or camera index')
    parser.add_argument('--headless', action='store_true',
                        help='Run detector without a GUI')
    parser.add_argument('--speed-limit', type=float,
                        default=30.0, help='Speed limit in km/h')

    args = parser.parse_args()

    flask_proc = multiprocessing.Process(
        target=_run_flask, args=(args.host, args.port), daemon=True)
    det_proc = multiprocessing.Process(
        target=_run_detector,
        args=(args.video, args.headless, args.speed_limit),
        daemon=True,
    )

    flask_proc.start()
    print(f"Flask server started (pid={flask_proc.pid})")
    time.sleep(1)
    det_proc.start()
    print(f"Detector started (pid={det_proc.pid})")

    def _terminate(_signum, _frame):
        print('Shutting down processes...')
        for p in (det_proc, flask_proc):
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        sys.exit(0)

    signal.signal(signal.SIGINT, _terminate)
    signal.signal(signal.SIGTERM, _terminate)

    try:
        while True:
            if not flask_proc.is_alive() and not det_proc.is_alive():
                break
            time.sleep(1)
    except KeyboardInterrupt:
        _terminate(None, None)


if __name__ == '__main__':
    main()
