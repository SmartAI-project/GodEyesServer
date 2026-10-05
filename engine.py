# GodEyes Engine v34 • 40-track session lock • 5s head-turn • 70% Face ID gate • 30 FPS preview
from pathlib import Path
import math
import os
import threading
import time
import traceback
import re
import socket
import uuid
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


@dataclass
class Detection:
    box: tuple
    person_score: float
    face: Optional[tuple] = None
    face_feature: Optional[np.ndarray] = None
    appearance_feature: Optional[np.ndarray] = None
    face_side: str = 'UNKNOWN'
    head_dir: float = 0.0
    head_yaw_deg: float = 0.0
    head_quality: float = 0.0
    mouth_score: float = 0.0
    mouth_motion: float = 0.0
    mouth_activity: float = 0.0
    face_y_ratio: float = 0.0
    object_types: tuple = ()
    object_scores: tuple = ()
    lower_motion: float = 0.0
    hand_motion: float = 0.0
    task_activity: float = 0.0

def _extract_ipv4_addresses(value: str):
    if not value:
        return []
    found = []
    for match in re.findall(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)", value):
        try:
            parts = [int(x) for x in match.split('.')]
        except ValueError:
            continue
        if all(0 <= p <= 255 for p in parts) and match not in found:
            found.append(match)
    return found


def _local_ipv4_addresses():
    """Return usable IPv4 addresses assigned to this Windows PC."""
    values = []
    try:
        host = socket.gethostname()
        infos = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_DGRAM)
        for item in infos:
            ip = str(item[4][0])
            if ip.startswith('127.') or ip.startswith('169.254.'):
                continue
            if ip not in values:
                values.append(ip)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.25)
        s.connect(('192.0.2.1', 9))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith(('127.', '169.254.')) and ip not in values:
            values.insert(0, ip)
    except Exception:
        pass
    return values


def _scan_tcp_port_on_local_subnets(port: int, timeout_per_host: float = 0.10, max_results: int = 16):
    """Bounded /24 scan on networks actually attached to this PC."""
    import concurrent.futures
    hosts = []
    seen = set()
    for local_ip in _local_ipv4_addresses():
        parts = local_ip.split('.')
        if len(parts) != 4:
            continue
        prefix = '.'.join(parts[:3])
        for last in range(1, 255):
            ip = f'{prefix}.{last}'
            if ip == local_ip or ip in seen:
                continue
            seen.add(ip)
            hosts.append(ip)

    if not hosts:
        return []

    def probe(ip):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(float(timeout_per_host))
        try:
            return ip if sock.connect_ex((ip, int(port))) == 0 else None
        except OSError:
            return None
        finally:
            try:
                sock.close()
            except Exception:
                pass

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=64, thread_name_prefix='GodEyesNetProbe') as pool:
        future_map = {pool.submit(probe, ip): ip for ip in hosts}
        for future in concurrent.futures.as_completed(future_map):
            try:
                ip = future.result()
            except Exception:
                ip = None
            if ip:
                results.append(ip)
                if len(results) >= int(max_results):
                    break
    return results


def discover_onvif_cameras(timeout: float = 2.0, max_results: int = 10):
    """Discover ONVIF cameras using multicast first, then local port scanning.

    Windows firewalls, AP isolation and some Wi-Fi routers can block WS-Discovery
    multicast. Tapo publishes ONVIF on TCP 2020, so the bounded local scan is a
    reliable fallback without exposing an IP-address field to the teacher.
    """
    results = []
    seen = set()

    message_id = f"uuid:{uuid.uuid4()}"
    probe = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
        'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
        'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
        'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
        '<e:Header>'
        f'<w:MessageID>{message_id}</w:MessageID>'
        '<w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
        '<w:Action mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>'
        '</e:Header>'
        '<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>'
        '</e:Envelope>'
    ).encode('utf-8')

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.settimeout(0.25)
        try:
            sock.sendto(probe, ('239.255.255.250', 3702))
        except OSError:
            pass
        end_time = time.time() + max(0.4, float(timeout))
        while time.time() < end_time and len(results) < int(max_results):
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            payload = data.decode('utf-8', errors='ignore')
            candidates = _extract_ipv4_addresses(payload)
            if not candidates and addr:
                candidates = [addr[0]]
            for ip in candidates:
                if ip.startswith(('127.', '169.254.')) or ip in seen:
                    continue
                seen.add(ip)
                results.append({'ip': ip, 'onvif_port': 2020, 'discovery': 'ws-discovery'})
                if len(results) >= int(max_results):
                    break
    finally:
        try:
            sock.close()
        except Exception:
            pass

    if len(results) < int(max_results):
        for ip in _scan_tcp_port_on_local_subnets(2020, timeout_per_host=0.10, max_results=max_results):
            if ip not in seen:
                seen.add(ip)
                results.append({'ip': ip, 'onvif_port': 2020, 'discovery': 'local-port-scan'})
                if len(results) >= int(max_results):
                    break
    return results


def probe_rtsp_camera(ip: str, username: str, password: str, stream: str = 'stream1', port: int = 554, timeout_ms: int = 2500):
    """Open one RTSP camera with the supplied camera-account credentials.

    Returns metadata when one frame can be decoded, otherwise None.
    """
    ip = str(ip or '').strip()
    username = str(username or '').strip()
    password = str(password or '')
    stream = str(stream or 'stream1').strip().lower()
    if not ip or not username or not password or stream not in ('stream1', 'stream2'):
        return None
    from urllib.parse import quote as _quote
    url = f'rtsp://{_quote(username, safe="")}:{_quote(password, safe="")}@{ip}:{int(port)}/{stream}'
    old_options = os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS')
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000'
    cap = None
    try:
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_ms),
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout_ms),
            ])
        except Exception:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG if hasattr(cv2, 'CAP_FFMPEG') else cv2.CAP_ANY)
        if cap is None or not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        return {'ip': ip, 'port': int(port), 'stream': stream, 'width': int(w), 'height': int(h)}
    except Exception:
        return None
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if old_options is None:
            os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS', None)
        else:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = old_options


class CameraWorker(QThread):

    frame_ready = Signal(object)
    state = Signal(str)

    def __init__(self, camera_index=0, source_type='WEBCAM', rtsp_url=''):
        super().__init__()
        self.camera_index = int(camera_index)
        self.source_type = str(source_type or 'WEBCAM').upper()
        self.rtsp_url = str(rtsp_url or '').strip()
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self.cap = None
        self.backend_name = ''
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0
        self._latest_frame = None
        self._latest_frame_time = 0.0
        self._last_ui_emit = 0.0

    def get_latest_frame(self, copy=True):
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy() if copy else self._latest_frame

    def latest_age(self):
        with self._lock:
            if not self._latest_frame_time:
                return None
            return max(0.0, time.time() - self._latest_frame_time)

    def _release(self):
        with self._lock:
            cap = self.cap
            self.cap = None
            self._latest_frame = None
            self._latest_frame_time = 0.0
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _open_webcam(self):
        # For an explicitly selected external USB webcam, do not blindly reuse
        # index 0 because Windows commonly assigns the built-in laptop camera
        # to index 0. Prefer a configured non-zero index; otherwise probe 1..9
        # first and only fall back to index 0 if no other camera opens.
        if self.source_type in {'USB', 'USB_WEBCAM'} and int(self.camera_index) == 0:
            indices = list(range(1, 10)) + [0]
        else:
            indices = [int(self.camera_index)]

        candidates = [
            (cv2.CAP_DSHOW, 'DIRECTSHOW'),
            (cv2.CAP_MSMF, 'MEDIA FOUNDATION'),
            (cv2.CAP_ANY, 'AUTO'),
        ]
        for camera_index in indices:
            for backend, name in candidates:
                if self._stop_event.is_set():
                    return None
                cap = None
                try:
                    cap = cv2.VideoCapture(camera_index, backend)
                    if not cap.isOpened():
                        if cap is not None:
                            cap.release()
                        continue
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    for key, value in [
                        (cv2.CAP_PROP_FRAME_WIDTH, 1280),
                        (cv2.CAP_PROP_FRAME_HEIGHT, 720),
                        (cv2.CAP_PROP_FPS, 30),
                    ]:
                        try:
                            cap.set(key, value)
                        except Exception:
                            pass
                    try:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    except Exception:
                        pass
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size:
                        self.camera_index = camera_index
                        self.actual_height, self.actual_width = frame.shape[:2]
                        try:
                            self.actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                        except Exception:
                            self.actual_fps = 0.0
                        self.backend_name = name
                        self._set_latest(frame)
                        return cap
                except Exception:
                    pass
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        return None

    def _open_rtsp(self):
        if not self.rtsp_url:
            self.state.emit('CAMERA ERROR • RTSP URL MISSING')
            return None
        cap = None
        # FFmpeg low-latency hints. They are process-local environment settings
        # used by OpenCV's FFmpeg backend when it is available.
        old_options = os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS')
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
            'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000'
        )
        try:
            backends = []
            if hasattr(cv2, 'CAP_FFMPEG'):
                backends.append((cv2.CAP_FFMPEG, 'FFMPEG'))
            backends.append((cv2.CAP_ANY, 'AUTO'))
            for backend, name in backends:
                if self._stop_event.is_set():
                    return None
                try:
                    try:
                        if name == 'FFMPEG':
                            cap = cv2.VideoCapture(
                                self.rtsp_url,
                                backend,
                                [
                                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000,
                                ],
                            )
                        else:
                            cap = cv2.VideoCapture(self.rtsp_url, backend)
                    except Exception:
                        cap = cv2.VideoCapture(self.rtsp_url, backend)
                    if not cap.isOpened():
                        if cap is not None:
                            cap.release()
                        cap = None
                        continue
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    try:
                        cap.set(cv2.CAP_PROP_FPS, 20)
                    except Exception:
                        pass
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size:
                        self.actual_height, self.actual_width = frame.shape[:2]
                        try:
                            self.actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                        except Exception:
                            self.actual_fps = 0.0
                        self.backend_name = f'RTSP {name}'
                        self._set_latest(frame)
                        return cap
                except Exception:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
            return None
        finally:
            if old_options is None:
                os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS', None)
            else:
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = old_options

    def _set_latest(self, frame):
        if frame is None or getattr(frame, 'size', 0) == 0:
            return
        with self._lock:
            self._latest_frame = frame.copy()
            self._latest_frame_time = time.time()

    def _open(self):
        if self.source_type in {'RTSP', 'WIFI_CAMERA', 'TAPO', 'TAPO_C230', 'TAPO_C230_RTSP'}:
            return self._open_rtsp()
        return self._open_webcam()

    def _source_label(self):
        if self.source_type in {'RTSP', 'WIFI_CAMERA', 'TAPO', 'TAPO_C230', 'TAPO_C230_RTSP'}:
            return 'WI-FI CAMERA'
        if self.source_type in {'USB', 'USB_WEBCAM'}:
            return f'USB WEBCAM #{self.camera_index}'
        return f'WEBCAM #{self.camera_index}'

    def run(self):
        self._stop_event.clear()
        self._release()
        self.state.emit(f'CONNECTING • {self._source_label()}')
        reconnect_delay = 0.25
        while not self._stop_event.is_set():
            cap = self._open()
            if cap is None:
                if self._stop_event.is_set():
                    break
                self.state.emit(f'CAMERA ERROR • CANNOT OPEN • {self._source_label()}')
                time.sleep(min(2.0, reconnect_delay))
                reconnect_delay = min(2.0, reconnect_delay * 1.5)
                continue
            reconnect_delay = 0.25
            with self._lock:
                self.cap = cap
            self.state.emit(
                f'CONNECTED • {self._source_label()} • '
                f'{self.actual_width}x{self.actual_height} • {self.backend_name}'
            )
            bad_reads = 0
            self._last_ui_emit = 0.0
            while not self._stop_event.is_set():
                with self._lock:
                    active_cap = self.cap
                if active_cap is None:
                    break
                try:
                    ok, frame = active_cap.read()
                except Exception:
                    ok, frame = False, None
                if not ok or frame is None or frame.size == 0:
                    bad_reads += 1
                    if bad_reads >= 5:
                        self.state.emit(f'CAMERA WARNING • {self._source_label()} • RECONNECTING')
                        break
                    time.sleep(0.02)
                    continue
                bad_reads = 0
                self._set_latest(frame)
                now = time.time()
                # UI preview is capped; the capture loop still drains the RTSP
                # stream as quickly as OpenCV delivers it, keeping only the newest frame.
                if now - self._last_ui_emit >= (1.0 / 30.0):
                    self._last_ui_emit = now
                    self.frame_ready.emit(frame)
            self._release()
            if not self._stop_event.is_set():
                self.state.emit(f'RECONNECTING • {self._source_label()}')
                time.sleep(reconnect_delay)
        self._release()
        self.state.emit('DISCONNECTED')

    def stop(self):
        self._stop_event.set()
        self._release()


class PersonTrack:
    """Single-person track with prediction + smooth box follow.

    The detector can move a bounding box abruptly between frames.  This class
    keeps a filtered box, a short velocity estimate, and the raw detector box
    so the UI follows the person incrementally instead of jumping.
    """

    def __init__(self, track_id, detection, now=None):
        now = time.time() if now is None else float(now)
        self.track_id = int(track_id)
        raw = tuple(float(v) for v in detection.box)
        self.raw_box = raw
        self.box = raw
        self.prev_box = raw
        self.hits = 1
        self.missed = 0
        self.age = 1
        self.last_seen = now
        self.first_seen = now
        self.last_update_time = now
        self.person_score = float(detection.person_score)
        self.face = detection.face
        self.face_feature = detection.face_feature
        self.appearance_feature = detection.appearance_feature
        self.face_side = detection.face_side
        self.head_dir = float(detection.head_dir)
        self.head_yaw_deg = float(detection.head_yaw_deg)
        self.head_quality = float(detection.head_quality)
        self.mouth_score = float(detection.mouth_score)
        self.mouth_motion = float(detection.mouth_motion)
        self.mouth_activity = float(detection.mouth_activity)
        self.face_y_ratio = float(detection.face_y_ratio)
        self.object_types = tuple(detection.object_types or ())
        self.object_scores = tuple(float(v) for v in (detection.object_scores or ()))
        self.lower_motion = float(detection.lower_motion)
        self.hand_motion = float(detection.hand_motion)
        self.task_activity = float(detection.task_activity)
        self.velocity = [0.0, 0.0, 0.0, 0.0]
        self.raw_center_history = [self.center]
        self.center_history = [self.center]
        self.face_center_history = []
        if self.face is not None:
            self.face_center_history.append(self._face_center(self.face))
        self.height_history = [self.box[3]]
        self.width_history = [self.box[2]]

    @staticmethod
    def _face_center(face):
        try:
            x, y, w, h = [float(v) for v in face[:4]]
            return (x + 0.5 * w, y + 0.5 * h)
        except Exception:
            return None

    @property
    def center(self):
        x, y, w, h = self.box
        return (x + 0.5 * w, y + 0.5 * h)

    @property
    def normalized_center(self):
        return self.center

    @property
    def stability(self):
        return min(1.0, self.hits / 6.0) * max(0.0, 1.0 - min(1.0, self.missed / 12.0))

    @property
    def confirmed(self):
        return self.hits >= 2 and self.age >= 2

    @property
    def visible(self):
        return self.missed == 0

    @staticmethod
    def _blend_box(a, b, alpha_pos=0.60, alpha_size=0.38):
        ax, ay, aw, ah = [float(v) for v in a]
        bx, by, bw, bh = [float(v) for v in b]
        return (
            ax + alpha_pos * (bx - ax),
            ay + alpha_pos * (by - ay),
            max(8.0, aw + alpha_size * (bw - aw)),
            max(8.0, ah + alpha_size * (bh - ah)),
        )

    def _predicted_box(self, horizon=1.0):
        x, y, w, h = self.box
        vx, vy, vw, vh = self.velocity
        horizon = max(0.0, min(3.0, float(horizon)))
        return (
            x + vx * horizon,
            y + vy * horizon,
            max(8.0, w + vw * horizon),
            max(8.0, h + vh * horizon),
        )

    def update(self, detection, now=None, protected=False):
        now = time.time() if now is None else float(now)
        raw = tuple(float(v) for v in detection.box)
        previous_box = self.box
        previous_center = self.center
        previous_time = self.last_update_time
        dt = max(1e-3, now - previous_time)

        # Fuse the detector's current box with a short forward prediction.
        # A protected/locked track gets a slightly stronger correction so it
        # follows the student promptly while still avoiding box jitter.
        predicted = self._predicted_box(1.0)
        prediction_weight = 0.14 if not protected else 0.10
        fused = tuple(
            (1.0 - prediction_weight) * raw[i] + prediction_weight * predicted[i]
            for i in range(4)
        )

        # Incremental follow: never jump directly to the detector box.
        alpha_pos = 0.58 if not protected else 0.72
        alpha_size = 0.36 if not protected else 0.48
        smooth = self._blend_box(
            previous_box,
            fused,
            alpha_pos=alpha_pos,
            alpha_size=alpha_size,
        )

        self.prev_box = previous_box
        self.raw_box = raw
        self.box = smooth
        self.hits += 1
        self.missed = 0
        self.age += 1
        self.last_seen = now
        self.last_update_time = now

        new_cx, new_cy = self.center
        old_cx, old_cy = previous_center
        raw_vx = (new_cx - old_cx) / dt
        raw_vy = (new_cy - old_cy) / dt
        raw_vw = (smooth[2] - previous_box[2]) / dt
        raw_vh = (smooth[3] - previous_box[3]) / dt
        # Velocity is filtered separately so prediction remains stable.
        self.velocity[0] = 0.62 * self.velocity[0] + 0.38 * raw_vx * 0.12
        self.velocity[1] = 0.62 * self.velocity[1] + 0.38 * raw_vy * 0.12
        self.velocity[2] = 0.70 * self.velocity[2] + 0.30 * raw_vw * 0.08
        self.velocity[3] = 0.70 * self.velocity[3] + 0.30 * raw_vh * 0.08

        self.person_score = 0.80 * self.person_score + 0.20 * float(detection.person_score)
        self.face = detection.face
        if detection.face_feature is not None:
            self.face_feature = detection.face_feature
        if detection.appearance_feature is not None:
            self.appearance_feature = detection.appearance_feature
        self.face_side = detection.face_side
        self.head_dir = float(detection.head_dir)
        self.head_yaw_deg = float(detection.head_yaw_deg)
        self.head_quality = float(detection.head_quality)
        self.mouth_score = float(detection.mouth_score)
        self.mouth_motion = float(detection.mouth_motion)
        self.mouth_activity = float(detection.mouth_activity)
        self.face_y_ratio = float(detection.face_y_ratio)
        self.object_types = tuple(detection.object_types or ())
        self.object_scores = tuple(float(v) for v in (detection.object_scores or ()))
        self.lower_motion = float(detection.lower_motion)
        self.hand_motion = float(detection.hand_motion)
        self.task_activity = float(detection.task_activity)

        self.raw_center_history.append(self._raw_center(raw))
        self.center_history.append(self.center)
        face_center = self._face_center(self.face)
        if face_center is not None:
            self.face_center_history.append(face_center)
        if len(self.raw_center_history) > 30:
            self.raw_center_history.pop(0)
        if len(self.center_history) > 30:
            self.center_history.pop(0)
        if len(self.face_center_history) > 30:
            self.face_center_history.pop(0)
        self.height_history.append(self.box[3])
        self.width_history.append(self.box[2])
        if len(self.height_history) > 30:
            self.height_history.pop(0)
        if len(self.width_history) > 30:
            self.width_history.pop(0)

    @staticmethod
    def _raw_center(box):
        x, y, w, h = [float(v) for v in box]
        return (x + 0.5 * w, y + 0.5 * h)

    def miss(self, now=None, protected=False):
        now = time.time() if now is None else float(now)
        self.missed += 1
        self.age += 1
        self.last_seen = now
        self.last_update_time = now

        # During a short detector gap, move the box with the person's latest
        # velocity rather than freezing it.  The step is bounded by box height.
        x, y, w, h = self.box
        vx, vy, vw, vh = self.velocity
        max_step = max(18.0, h * (0.78 if protected else 0.60))
        step_x = float(np.clip(vx, -max_step, max_step))
        step_y = float(np.clip(vy, -max_step, max_step))
        self.prev_box = self.box
        self.box = (
            x + step_x,
            y + step_y,
            max(8.0, w + float(np.clip(vw, -0.08 * w, 0.08 * w))),
            max(8.0, h + float(np.clip(vh, -0.08 * h, 0.08 * h))),
        )
        decay = 0.88 if protected else 0.82
        self.velocity = [v * decay for v in self.velocity]
        self.center_history.append(self.center)
        if len(self.center_history) > 30:
            self.center_history.pop(0)
        self.height_history.append(self.box[3])
        self.width_history.append(self.box[2])
        if len(self.height_history) > 30:
            self.height_history.pop(0)
        if len(self.width_history) > 30:
            self.width_history.pop(0)


class PersonTracker:
    MAX_TRACKS = 40
    """Person tracker optimized for a locked classroom student.

    Matching uses IoU, center/predicted position, appearance, and—when both
    boxes contain a face—the face anchor.  Locked tracks receive a stronger
    prediction/face weight and a wider temporary gate to keep the same track
    while the student walks, turns, or is briefly occluded.
    """

    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.max_missed = 90
        self.max_missed_protected = 180
        self.max_assignment_distance = 2.35
        self.min_assignment_score = 0.21
        self.protected_track_ids = set()

    def set_protected_track_ids(self, track_ids):
        self.protected_track_ids = {int(v) for v in (track_ids or [])}

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2 = ax + aw
        ay2 = ay + ah
        bx2 = bx + bw
        by2 = by + bh
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = max(1.0, aw * ah + bw * bh - inter)
        return inter / union

    @staticmethod
    def _center_from_box(box):
        x, y, w, h = box
        return x + w * 0.5, y + h * 0.5

    @staticmethod
    def _center_distance(a, b):
        ax, ay = PersonTracker._center_from_box(a)
        bx, by = PersonTracker._center_from_box(b)
        ah = max(1.0, float(a[3]))
        bh = max(1.0, float(b[3]))
        d = math.hypot(ax - bx, ay - by)
        scale = max(60.0, 0.50 * (ah + bh))
        return min(3.0, d / scale)

    @staticmethod
    def _appearance_distance(a, b):
        if a is None or b is None:
            return 1.0
        try:
            aa = np.asarray(a, dtype=np.float32).ravel()
            bb = np.asarray(b, dtype=np.float32).ravel()
            if aa.size != bb.size or aa.size == 0:
                return 1.0
            denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
            if denom <= 1e-8:
                return 1.0
            similarity = float(np.dot(aa, bb) / denom)
            similarity = max(-1.0, min(1.0, similarity))
            return float(1.0 - 0.5 * (similarity + 1.0))
        except Exception:
            return 1.0

    @staticmethod
    def _face_embedding_similarity(a, b):
        if a is None or b is None:
            return -1.0
        try:
            aa = np.asarray(a, dtype=np.float32).ravel()
            bb = np.asarray(b, dtype=np.float32).ravel()
            if aa.size == 0 or aa.size != bb.size:
                return -1.0
            denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
            if denom <= 1e-8:
                return -1.0
            return float(np.dot(aa, bb) / denom)
        except Exception:
            return -1.0

    @staticmethod
    def _face_distance(a, b, scale_height=120.0):
        if a is None or b is None:
            return 1.0
        try:
            ax, ay, aw, ah = [float(v) for v in a[:4]]
            bx, by, bw, bh = [float(v) for v in b[:4]]
            acx, acy = ax + aw * 0.5, ay + ah * 0.5
            bcx, bcy = bx + bw * 0.5, by + bh * 0.5
            d = math.hypot(acx - bcx, acy - bcy)
            return float(min(3.0, d / max(40.0, float(scale_height) * 0.55)))
        except Exception:
            return 1.0

    @staticmethod
    def _repair_detection_box(detection):
        """Keep the tracked person box attached to its detected face.

        Some person detections can be shifted sideways relative to the face.
        When that happens, shift the box toward the face anchor before matching
        and smoothing.  This directly prevents a locked box from staying on a
        nearby chair/background region.
        """
        face = getattr(detection, 'face', None)
        if face is None:
            return detection
        try:
            bx, by, bw, bh = [float(v) for v in detection.box]
            fx, fy, fw, fh = [float(v) for v in face[:4]]
            fcx = fx + 0.5 * fw
            fcy = fy + 0.5 * fh
            inside = (bx <= fcx <= bx + bw) and (by <= fcy <= by + bh)
            if inside:
                return detection

            # Desired placement: face centered horizontally and in the upper
            # quarter of the person box.  Move mostly, but not instantaneously.
            desired_x = fcx - 0.50 * bw
            desired_y = fcy - 0.20 * bh
            repaired_x = 0.20 * bx + 0.80 * desired_x
            repaired_y = 0.20 * by + 0.80 * desired_y
            detection.box = (
                int(round(max(0.0, repaired_x))),
                int(round(max(0.0, repaired_y))),
                int(round(max(8.0, bw))),
                int(round(max(8.0, bh))),
            )
        except Exception:
            return detection
        return detection

    def _pair_score(self, track, detection):
        iou = self._iou(track.box, detection.box)
        distance = self._center_distance(track.box, detection.box)
        appearance = self._appearance_distance(track.appearance_feature, detection.appearance_feature)
        face_distance = self._face_distance(
            getattr(track, 'face', None),
            getattr(detection, 'face', None),
            scale_height=max(60.0, float(track.box[3])),
        )
        face_score = 1.0 - min(1.0, face_distance)
        face_embedding_similarity = self._face_embedding_similarity(
            getattr(track, 'face_feature', None),
            getattr(detection, 'face_feature', None),
        )
        face_embedding_score = max(0.0, min(1.0, face_embedding_similarity)) if face_embedding_similarity >= 0.0 else 0.0

        current_cx, current_cy = self._center_from_box(track.box)
        next_cx, next_cy = self._center_from_box(detection.box)
        predicted_distance = distance
        velocity_bonus = 0.0

        if len(track.center_history) >= 2:
            px, py = track.center_history[-2]
            cx, cy = track.center_history[-1]
            vx = cx - px
            vy = cy - py
            horizon = min(3, max(1, int(track.missed) + 1))
            pred = (current_cx + vx * horizon, current_cy + vy * horizon)
            pd_px = math.hypot(pred[0] - next_cx, pred[1] - next_cy)
            predicted_distance = min(3.0, pd_px / max(70.0, track.box[3] * 0.95))
            velocity_bonus = max(0.0, 1.0 - min(1.0, predicted_distance))

        position_score = 1.0 - min(1.0, min(distance, predicted_distance))
        protected = track.track_id in self.protected_track_ids
        if protected and face_distance < 3.0:
            score = (
                0.20 * iou
                + 0.23 * position_score
                + 0.14 * (1.0 - appearance)
                + 0.18 * face_score
                + 0.20 * face_embedding_score
                + 0.05 * velocity_bonus
            )
            # A protected identity must not silently jump to a different
            # visible face. Only the tracker association is allowed to fail;
            # IdentityLock will then keep the student uncertain until a strong
            # embedding match re-establishes the identity.
            if face_embedding_similarity >= 0.0 and face_embedding_similarity < 0.40:
                return 0.0
        else:
            score = (
                0.38 * iou
                + 0.30 * position_score
                + 0.20 * (1.0 - appearance)
                + 0.07 * face_score
                + 0.05 * velocity_bonus
            )

        gate = self.max_assignment_distance
        if protected:
            gate += min(0.80, 0.22 * float(track.missed))
        else:
            gate += min(0.55, 0.16 * float(track.missed))

        if iou < 0.02 and min(distance, predicted_distance) > gate and appearance > 0.78:
            return 0.0
        if protected and getattr(track, 'face', None) is not None and getattr(detection, 'face', None) is not None:
            # A locked student with a visible face should not jump to a far-away
            # face simply because its person boxes overlap poorly.
            if face_distance > 1.75 and distance > 0.95 and appearance > 0.72:
                return 0.0
        return score

    def update(self, detections, now=None):
        now = time.time() if now is None else float(now)
        detections = [self._repair_detection_box(d) for d in list(detections or [])]
        active_ids = list(self.tracks.keys())
        pairs = []

        for tid in active_ids:
            track = self.tracks[tid]
            for di, detection in enumerate(detections):
                score = self._pair_score(track, detection)
                if score <= 0.0:
                    continue
                iou = self._iou(track.box, detection.box)
                dist = self._center_distance(track.box, detection.box)
                pred_dist = dist
                if len(track.center_history) >= 2:
                    px, py = track.center_history[-2]
                    cx, cy = track.center_history[-1]
                    vx = cx - px
                    vy = cy - py
                    nx, ny = self._center_from_box(detection.box)
                    horizon = min(3, max(1, int(track.missed) + 1))
                    pred = (cx + vx * horizon, cy + vy * horizon)
                    pred_dist = min(3.0, math.hypot(pred[0] - nx, pred[1] - ny) / max(70.0, track.box[3] * 0.95))
                gate = self.max_assignment_distance
                if tid in self.protected_track_ids:
                    gate += min(0.80, 0.22 * float(track.missed))
                else:
                    gate += min(0.55, 0.16 * float(track.missed))
                if iou >= 0.02 or min(dist, pred_dist) <= gate:
                    pairs.append((score, tid, di))

        pairs.sort(key=lambda item: item[0], reverse=True)
        matched_tracks = set()
        matched_detections = set()
        for score, tid, di in pairs:
            if tid in matched_tracks or di in matched_detections:
                continue
            if score < self.min_assignment_score:
                continue
            protected = tid in self.protected_track_ids
            self.tracks[tid].update(detections[di], now, protected=protected)
            matched_tracks.add(tid)
            matched_detections.add(di)

        for tid in list(active_ids):
            if tid not in matched_tracks and tid in self.tracks:
                protected = tid in self.protected_track_ids
                self.tracks[tid].miss(now, protected=protected)
                max_missed = self.max_missed_protected if protected else self.max_missed
                if self.tracks[tid].missed > max_missed:
                    del self.tracks[tid]
                    self.protected_track_ids.discard(tid)

        for di, detection in enumerate(detections):
            if di in matched_detections:
                continue
            if len(self.tracks) >= self.MAX_TRACKS:
                # Never exceed the classroom track budget. Prefer preserving
                # already protected identities and confirmed tracks.
                removable = [
                    t for t in self.tracks.values()
                    if t.track_id not in self.protected_track_ids
                ]
                if not removable:
                    removable = list(self.tracks.values())
                victim = min(
                    removable,
                    key=lambda t: (
                        int(t.track_id in self.protected_track_ids),
                        int(t.confirmed),
                        -int(t.missed),
                        int(t.hits),
                    )
                )
                self.tracks.pop(victim.track_id, None)
                self.protected_track_ids.discard(victim.track_id)
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = PersonTrack(tid, detection, now)

        # A defensive hard cap keeps both memory and rendering bounded.
        if len(self.tracks) > self.MAX_TRACKS:
            ranked = sorted(
                self.tracks.values(),
                key=lambda t: (
                    int(t.track_id in self.protected_track_ids),
                    int(t.confirmed),
                    -int(t.missed),
                    int(t.hits),
                ),
                reverse=True,
            )
            keep_ids = {t.track_id for t in ranked[:self.MAX_TRACKS]}
            for tid in list(self.tracks):
                if tid not in keep_ids:
                    self.tracks.pop(tid, None)
                    self.protected_track_ids.discard(tid)

        return list(self.tracks.values())

    def confirmed_visible(self):
        return [t for t in self.tracks.values() if t.confirmed and t.visible]

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.protected_track_ids.clear()


class IdentityLock:
    """
    Identity manager for classroom monitoring.

    Important design rule:
    - Face ID is used when a usable face embedding is available.
    - Once a person has been identified, the identity is held while the same
      person track continues even if the face temporarily disappears.
    - If the tracker creates a new track after a temporary posture/occlusion
      change, a conservative appearance + position re-identification step can
      recover the previously known student.

    This prevents a student who bends their head down from immediately becoming
    UNKNOWN, without pretending that a hidden face can be re-identified from
    the face model alone.
    """

    def __init__(self):
        self.locked = False
        self.track_to_student = {}
        self.student_profiles = {}
        self.student_to_track = {}
        self.next_student = 1

        # Legacy/local track-reidentification settings.
        self.max_match_distance = 0.78
        self.min_reid_score = 0.49

        # Server Face ID roster.
        self.server_roster = {}
        self.server_threshold = 0.70
        self.server_margin = 0.08

        # Temporary identity continuity when face disappears.
        # The lock is meant to follow a student for the whole monitoring
        # session, not only for a few seconds after the last face frame.
        self.hold_identity_seconds = float('inf')
        self.reid_threshold = 0.84
        self.reid_margin = 0.10
        self.reid_appearance_max_distance = 0.36
        self.reid_position_max_distance = 1.00

        # Snapshot of tracks visible in the current processed frame. It lets
        # face matching distinguish a truly lost old track from an identity
        # that is still actively visible and must not be stolen.
        self.active_track_status = {}

    @staticmethod
    def _cosine_similarity(a, b):
        if a is None or b is None:
            return -1.0
        try:
            aa = np.asarray(a, dtype=np.float32).ravel()
            bb = np.asarray(b, dtype=np.float32).ravel()
            if aa.size == 0 or aa.size != bb.size:
                return -1.0
            denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
            if denom <= 1e-8:
                return -1.0
            return float(np.dot(aa, bb) / denom)
        except Exception:
            return -1.0

    @staticmethod
    def _display_label(item):
        code = str(item.get('student_code') or '').strip()
        name = str(item.get('full_name') or '').strip()
        if code and name:
            return f'{code} • {name}'
        return name or code or f"STUDENT {item.get('id', '')}"

    def set_server_roster(self, roster):
        parsed = {}
        for item in list(roster or []):
            try:
                if not item.get('ready'):
                    continue
                raw = item.get('face_embedding')
                feature = np.asarray(raw, dtype=np.float32).ravel()
                if feature.size == 0:
                    continue
                norm = float(np.linalg.norm(feature))
                if norm <= 1e-8:
                    continue
                feature = feature / norm
                label = self._display_label(item)
                parsed[label] = {
                    'student_id': int(item.get('id') or 0),
                    'student_code': str(item.get('student_code') or ''),
                    'full_name': str(item.get('full_name') or ''),
                    'face_feature': feature,
                }
            except Exception:
                continue
        self.server_roster = parsed
        self.locked = False
        self.track_to_student.clear()
        self.student_profiles.clear()
        self.student_to_track.clear()
        self.active_track_status.clear()

    def begin_frame(self, tracks):
        """Tell IdentityLock which tracks are alive/visible in this frame."""
        status = {}
        for track in list(tracks or []):
            try:
                status[int(track.track_id)] = {
                    'missed': int(getattr(track, 'missed', 0)),
                    'confirmed': bool(getattr(track, 'confirmed', False)),
                    'visible': bool(getattr(track, 'visible', False)),
                }
            except Exception:
                continue
        self.active_track_status = status

    def student_id_for_label(self, label):
        profile = self.server_roster.get(str(label))
        return int(profile.get('student_id', 0)) if profile else 0

    @staticmethod
    def _feature_distance(a, b):
        if a is None or b is None:
            return 1.0
        try:
            aa = np.asarray(a, dtype=np.float32).ravel()
            bb = np.asarray(b, dtype=np.float32).ravel()
            if aa.size != bb.size or aa.size == 0:
                return 1.0
            denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
            if denom <= 1e-8:
                return 1.0
            sim = float(np.dot(aa, bb) / denom)
            sim = max(-1.0, min(1.0, sim))
            return 1.0 - 0.5 * (sim + 1.0)
        except Exception:
            return 1.0

    @staticmethod
    def _position_distance(a, b, frame_size=None):
        if a is None or b is None:
            return 1.0
        ax, ay = a
        bx, by = b
        if frame_size:
            fw, fh = frame_size
            dx = (ax - bx) / max(1.0, float(fw))
            dy = (ay - by) / max(1.0, float(fh))
            return float(min(1.0, math.hypot(dx, dy) * 2.2))
        return float(min(1.0, math.hypot(ax - bx, ay - by) / 500.0))

    def _order_tracks(self, tracks):
        ordered = [t for t in tracks if t.confirmed and t.visible]
        ordered.sort(key=lambda t: (round(t.center[1] / 60.0), t.center[0]))
        return ordered

    def lock(self, tracks, frame_size=None):
        visible = self._order_tracks(list(tracks))[:40]
        self.track_to_student.clear()
        self.student_profiles.clear()
        self.student_to_track.clear()
        self.next_student = 1
        now = time.time()

        if self.server_roster:
            assignments = []
            for track in visible:
                if track.face_feature is None:
                    continue
                best_label = None
                best_score = -1.0
                second_score = -1.0
                for label, item in self.server_roster.items():
                    score = self._cosine_similarity(
                        track.face_feature,
                        item.get('face_feature')
                    )
                    if score > best_score:
                        second_score = best_score
                        best_score = score
                        best_label = label
                    elif score > second_score:
                        second_score = score
                if (
                    best_label is not None
                    and best_score >= self.server_threshold
                    and best_score - max(-1.0, second_score) >= self.server_margin
                ):
                    assignments.append((best_score, track, best_label))

            assignments.sort(key=lambda item: item[0], reverse=True)
            used = set()
            for score, track, label in assignments:
                if label in used or track.track_id in self.track_to_student:
                    continue
                used.add(label)
                self._bind_identity(label, track, frame_size, now)

        else:
            for track in visible:
                sid = f'HS {self.next_student:02d}'
                self.next_student += 1
                self._bind_identity(sid, track, frame_size, now)

        self.locked = bool(self.track_to_student or self.student_profiles)
        return self.mapping()

    def _bind_identity(self, sid, track, frame_size=None, now=None):
        now = time.time() if now is None else now
        self.track_to_student[track.track_id] = sid
        self.student_to_track[sid] = track.track_id

        roster_item = self.server_roster.get(sid) if self.server_roster else None
        server_embedding = None
        if roster_item is not None:
            server_embedding = roster_item.get('face_feature')

        self.student_profiles[sid] = {
            'face_feature': (
                track.face_feature.copy()
                if track.face_feature is not None
                else server_embedding.copy() if server_embedding is not None else None
            ),
            'appearance_feature': (
                track.appearance_feature.copy()
                if track.appearance_feature is not None else None
            ),
            'server_embedding': (
                server_embedding.copy() if server_embedding is not None else None
            ),
            'center': track.center,
            'frame_size': frame_size,
            'last_seen': now,
            'last_face_seen': now if track.face_feature is not None else 0.0,
            'track_id': track.track_id,
            'student_id': int(roster_item['student_id']) if roster_item else 0,
        }

    def _profile_score(self, track, profile, frame_size=None):
        fd = self._feature_distance(
            track.face_feature,
            profile.get('face_feature')
        )
        ad = self._feature_distance(
            track.appearance_feature,
            profile.get('appearance_feature')
        )
        pd = self._position_distance(
            track.center,
            profile.get('center'),
            frame_size or profile.get('frame_size')
        )
        score = 1.0 - (0.30 * fd + 0.48 * ad + 0.22 * pd)
        return max(0.0, min(1.0, score)), fd, ad, pd

    def _appearance_reid_score(self, track, profile, frame_size=None):
        ad = self._feature_distance(
            track.appearance_feature,
            profile.get('appearance_feature')
        )
        pd = self._position_distance(
            track.center,
            profile.get('center'),
            frame_size or profile.get('frame_size')
        )

        # When a new person track is created, SFace is still the strongest
        # identity cue. Appearance + position then stabilize the hand-off.
        fd = self._feature_distance(
            track.face_feature,
            profile.get('face_feature')
        )
        if track.face_feature is not None and profile.get('face_feature') is not None:
            score = 1.0 - (0.55 * fd + 0.30 * ad + 0.15 * pd)
        else:
            score = 1.0 - (0.70 * ad + 0.30 * pd)
        return max(0.0, min(1.0, score)), ad, pd

    def _refresh_profile(self, sid, track, frame_size=None, identity_source='TRACK_HOLD'):
        profile = self.student_profiles.get(sid)
        if profile is None:
            return

        now = time.time()

        if track.face_feature is not None:
            old = profile.get('face_feature')
            anchor = profile.get('server_embedding') if profile.get('server_embedding') is not None else old
            agrees = True
            if anchor is not None:
                agrees = self._cosine_similarity(track.face_feature, anchor) >= self.server_threshold
            if agrees:
                if old is None:
                    profile['face_feature'] = track.face_feature.copy()
                else:
                    # Very slow update keeps the original enrollment anchor stable.
                    updated = 0.985 * old + 0.015 * track.face_feature
                    norm = float(np.linalg.norm(updated))
                    profile['face_feature'] = (
                        updated / norm if norm > 1e-8 else updated
                    )
                profile['last_face_seen'] = now

        if track.appearance_feature is not None:
            old = profile.get('appearance_feature')
            if old is None:
                profile['appearance_feature'] = track.appearance_feature.copy()
            else:
                updated = 0.99 * old + 0.01 * track.appearance_feature
                norm = float(np.linalg.norm(updated))
                profile['appearance_feature'] = (
                    updated / norm if norm > 1e-8 else updated
                )

        profile['center'] = track.center
        profile['frame_size'] = frame_size or profile.get('frame_size')
        profile['last_seen'] = now
        profile['track_id'] = track.track_id
        profile['identity_source'] = identity_source

        self.student_to_track[sid] = track.track_id
        self.track_to_student[track.track_id] = sid

    def _locked_face_similarity(self, sid, track):
        """Compare the current face against the locked student's anchors."""
        profile = self.student_profiles.get(sid) or {}
        candidates = []
        current = getattr(track, 'face_feature', None)
        if current is None:
            return -1.0
        stored = profile.get('face_feature')
        if stored is not None:
            candidates.append(self._cosine_similarity(current, stored))
        server = profile.get('server_embedding')
        if server is not None:
            candidates.append(self._cosine_similarity(current, server))
        values = [v for v in candidates if v >= -1.0]
        return max(values) if values else -1.0

    def _match_face_to_server(self, track):
        if not self.server_roster or track.face_feature is None:
            return None, -1.0, -1.0

        best_label = None
        best_score = -1.0
        second_score = -1.0

        for label, item in self.server_roster.items():
            assigned_track = self.student_to_track.get(label)
            if assigned_track is not None and assigned_track != track.track_id:
                assigned_status = self.active_track_status.get(assigned_track, {})
                # Only block reassignment when that identity is still attached
                # to a live confirmed track in the current frame.
                if assigned_status.get('visible') and assigned_status.get('confirmed') and not assigned_status.get('missed', 0):
                    continue

            score = self._cosine_similarity(
                track.face_feature,
                item.get('face_feature')
            )

            if score > best_score:
                second_score = best_score
                best_score = score
                best_label = label
            elif score > second_score:
                second_score = score

        return best_label, best_score, second_score

    def _recover_from_profile(self, track, frame_size=None):
        if not self.student_profiles:
            return None, 0.0

        now = time.time()
        candidates = []

        for sid, profile in self.student_profiles.items():
            # Session-long lock: identity memory does not expire by time.
            previous_track = self.student_to_track.get(sid)
            if previous_track == track.track_id:
                continue

            # Do not steal an identity from a track that is still confirmed and
            # visible. Re-acquisition is allowed once the old track is missing.
            previous_status = self.active_track_status.get(previous_track, {})
            if previous_status.get('visible') and previous_status.get('confirmed') and not previous_status.get('missed', 0):
                continue

            # Appearance-only recovery is permitted only when the face is
            # genuinely unavailable. The face mismatch path is handled in
            # resolve() and is intentionally blocked.
            if getattr(track, 'face_feature', None) is not None:
                continue

            score, appearance_distance, position_distance = self._appearance_reid_score(
                track,
                profile,
                frame_size,
            )

            if appearance_distance > self.reid_appearance_max_distance:
                continue
            # Position is a weak supporting signal, not a hard identity gate.
            if score < self.reid_threshold:
                continue

            candidates.append((
                score,
                sid,
                appearance_distance,
                position_distance,
            ))

        if not candidates:
            return None, 0.0

        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0]
        second = candidates[1][0] if len(candidates) > 1 else -1.0

        if len(candidates) > 1 and best[0] - second < self.reid_margin:
            return None, best[0]

        sid = best[1]
        previous_track = self.student_to_track.get(sid)

        if previous_track is not None and previous_track != track.track_id:
            self.track_to_student.pop(previous_track, None)

        self._refresh_profile(
            sid,
            track,
            frame_size,
            identity_source='TRACK_REID'
        )

        return sid, best[0]

    def resolve(self, track, frame_size=None):
        # 1. Existing LOCKED track.
        #    When the face returns, always re-verify the embedding against the
        #    enrolled roster. The old identity is NEVER allowed to silently
        #    switch to another student because of a track/box association.
        if track.track_id in self.track_to_student:
            sid = self.track_to_student[track.track_id]
            profile = self.student_profiles.get(sid)

            if profile is not None:
                if track.face_feature is not None:
                    # Re-match the returning face against the complete enrolled
                    # dataset first. Only the original locked label may resume.
                    best_label, best_score, second_score = self._match_face_to_server(track)
                    margin_ok = best_score - max(-1.0, second_score) >= self.server_margin

                    # If the roster match is absent/weak/ambiguous, keep the
                    # identity uncertain. Do not fall back to appearance.
                    if (
                        best_label is None
                        or best_score < self.server_threshold
                        or not margin_ok
                        or best_label != sid
                    ):
                        locked_sim = self._locked_face_similarity(sid, track)
                        return 'IDENTITY UNCERTAIN', max(0.0, min(1.0, max(best_score, locked_sim)))

                    self._refresh_profile(
                        sid,
                        track,
                        frame_size,
                        identity_source='FACE_REVERIFIED'
                    )
                    return sid, max(0.0, min(1.0, best_score))

                self._refresh_profile(
                    sid,
                    track,
                    frame_size,
                    identity_source='TRACK_HOLD'
                )
                return sid, 0.90

        # 2. Fresh face observation: lock only on a strong Face ID match.
        # A visible face that does NOT reach the 70% threshold is never allowed
        # to inherit somebody else's identity from appearance matching.
        if self.server_roster and track.face_feature is not None:
            best_label, best_score, second_score = self._match_face_to_server(track)
            margin_ok = best_score - max(-1.0, second_score) >= self.server_margin

            if best_label is not None and best_score >= self.server_threshold and margin_ok:
                previous_track = self.student_to_track.get(best_label)
                if previous_track is not None and previous_track != track.track_id:
                    previous_status = self.active_track_status.get(previous_track, {})
                    # Never steal a live identity.
                    if previous_status.get('visible') and previous_status.get('confirmed') and not previous_status.get('missed', 0):
                        return 'IDENTITY UNCERTAIN', 0.0
                    self.track_to_student.pop(previous_track, None)

                if best_label not in self.student_profiles:
                    self._bind_identity(best_label, track, frame_size)
                else:
                    self._refresh_profile(
                        best_label,
                        track,
                        frame_size,
                        identity_source='FACE'
                    )

                return best_label, max(0.0, min(1.0, best_score))

            # A visible but weak/mismatching embedding must stay uncertain.
            return 'IDENTITY UNCERTAIN', max(0.0, min(1.0, best_score))

        # 3. Face temporarily unavailable: recover a previously LOCKED identity
        # from the same person's appearance only. This path is intentionally
        # unavailable while a face embedding is visible, preventing cross-person
        # identity transfer when the current face contradicts the stored face.
        sid, reid_score = self._recover_from_profile(track, frame_size)
        if sid is not None:
            return sid, reid_score

        # 4. No prior identity yet.
        if not self.student_profiles:
            if self.server_roster:
                return 'IDENTITY UNCERTAIN', 0.0
            return (
                f'CANDIDATE {track.track_id:02d}',
                max(0.45, min(0.99, 0.55 + 0.40 * track.stability))
            )

        # There are known identities, but this track cannot be linked safely.
        return 'IDENTITY UNCERTAIN', 0.0

    def mapping(self):
        return dict(self.track_to_student)

    def reset(self):
        self.locked = False
        self.track_to_student.clear()
        self.student_profiles.clear()
        self.student_to_track.clear()
        self.next_student = 1

class SmartVision:
    MODEL_NAMES = ('yolo11n.pt', 'yolov8n.pt')

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model_path = self._find_model()
        self.model_source = ''
        self.model = None
        self.front = self._load_cascade('haarcascade_frontalface_default.xml')
        self.profile = self._load_cascade('haarcascade_profileface.xml')
        self.face_detector = None
        self.face_recognizer = None
        self.face_model_error = ''
        self._load_face_models()
        self.frame_index = 0
        self.previous_gray = None
        self.previous_frame = None
        self.previous_mouth = {}
        self.last_detections = []
        self.inference_count = 0
        self.last_inference_ms = 0.0
        self.last_model_error = ''
        self._load_yolo()

    def _load_face_models(self):
        # Prefer models bundled with the app; fall back to the shared local
        # GodEyesServer model directory used during development.
        candidates = [
            self.model_dir,
            Path(r'D:\GodEyesServer\data\face_models'),
            Path(r'D:\GodEyes\app\assets\models'),
        ]
        seen = set()
        for base in candidates:
            try:
                base = Path(base)
                key = str(base.resolve()).lower()
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            yunet_path = base / 'face_detection_yunet_2023mar.onnx'
            sface_path = base / 'face_recognition_sface_2021dec.onnx'
            if not (yunet_path.exists() and sface_path.exists()):
                continue
            try:
                if hasattr(cv2, 'FaceDetectorYN') and yunet_path.stat().st_size > 100000:
                    self.face_detector = cv2.FaceDetectorYN.create(str(yunet_path), '', (320, 320), 0.50, 0.30, 5000)
                if hasattr(cv2, 'FaceRecognizerSF') and sface_path.stat().st_size > 100000:
                    self.face_recognizer = cv2.FaceRecognizerSF.create(str(sface_path), '')
                if self.face_detector is not None and self.face_recognizer is not None:
                    return
            except Exception as exc:
                self.face_model_error = str(exc)
        self.face_detector = None
        self.face_recognizer = None

    def _yunet_faces(self, frame):
        if self.face_detector is None:
            return []
        try:
            h, w = frame.shape[:2]
            self.face_detector.setInputSize((int(w), int(h)))
            _, faces = self.face_detector.detect(frame)
            return list(faces) if faces is not None else []
        except Exception:
            return []

    @staticmethod
    def _select_yunet_face(faces, person_box):
        px, py, pw, ph = person_box
        candidates = []
        for face in faces:
            try:
                fx, fy, fw, fh = [float(v) for v in face[:4]]
            except Exception:
                continue
            cx = fx + fw * 0.5
            cy = fy + fh * 0.5
            if px <= cx <= px + pw and py <= cy <= py + ph * 0.72:
                score = fw * fh
                candidates.append((score, face))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _sface_feature(self, frame, face):
        if self.face_recognizer is None or face is None:
            return None

        features = []
        try:
            aligned = self.face_recognizer.alignCrop(frame, face)
            feature = self.face_recognizer.feature(aligned)
            feature = np.asarray(feature, dtype=np.float32).ravel()
            norm = float(np.linalg.norm(feature))
            if feature.size and norm > 1e-8:
                features.append(feature / norm)
        except Exception:
            pass

        # Low-light / mildly blurred faces get a second embedding from a
        # contrast-enhanced view. We combine the two only when both are valid;
        # this improves robustness without changing the identity metric.
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
            aligned2 = self.face_recognizer.alignCrop(enhanced, face)
            feature2 = self.face_recognizer.feature(aligned2)
            feature2 = np.asarray(feature2, dtype=np.float32).ravel()
            norm2 = float(np.linalg.norm(feature2))
            if feature2.size and norm2 > 1e-8:
                features.append(feature2 / norm2)
        except Exception:
            pass

        if not features:
            return None
        if len(features) == 1:
            return features[0]
        merged = np.mean(np.stack(features, axis=0), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(merged))
        return merged / norm if norm > 1e-8 else features[0]

    def _find_model(self):
        roots = [
            self.model_dir,
            self.model_dir / 'weights',
            self.model_dir.parent,
            Path(__file__).resolve().parent,
            Path(__file__).resolve().parent.parent,
            Path.cwd(),
        ]
        env_paths = [
            os.environ.get('FOCUSAI_MODEL', ''),
            os.environ.get('YOLO_MODEL_PATH', ''),
        ]
        candidates = []
        for raw in env_paths:
            if raw:
                candidates.append(Path(raw))
        for root in roots:
            for name in self.MODEL_NAMES:
                candidates.append(root / name)
        seen = set()
        for candidate in candidates:
            try:
                p = candidate.expanduser().resolve()
            except Exception:
                continue
            key = str(p).lower()
            if key in seen:
                continue
            seen.add(key)
            if p.exists() and p.is_file() and p.stat().st_size > 1024 * 1024:
                return p
        return None

    def _load_yolo(self):
        # YOLO is preferred, but the app must remain usable when the model file
        # is not installed yet. In that case SmartVision falls back to YuNet
        # face detections to create person tracks.
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.last_model_error = f'Ultralytics unavailable: {exc}'
            self.model = None
            self.model_source = 'YuNet Face Fallback'
            return

        if self.model_path is None:
            self.last_model_error = 'YOLO model not found; using YuNet Face Fallback.'
            self.model = None
            self.model_source = 'YuNet Face Fallback'
            return

        try:
            self.model = YOLO(str(self.model_path))
            self.model_source = str(self.model_path)
            try:
                self.model.fuse()
            except Exception:
                pass
        except Exception as exc:
            self.last_model_error = f'YOLO load failed: {exc}; using YuNet Face Fallback.'
            self.model = None
            self.model_source = 'YuNet Face Fallback'

    def _load_cascade(self, name):
        paths = [
            self.model_dir / name,
            Path(cv2.data.haarcascades) / name,
        ]
        for path in paths:
            try:
                if path.exists():
                    cascade = cv2.CascadeClassifier(str(path))
                    if not cascade.empty():
                        return cascade
            except Exception:
                pass
        return cv2.CascadeClassifier()

    @staticmethod
    def _clamp_box(box, width, height):
        x, y, w, h = [int(v) for v in box]
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        x2 = max(x + 1, min(width, x + w))
        y2 = max(y + 1, min(height, y + h))
        return x, y, x2 - x, y2 - y

    @staticmethod
    def _gray_feature(gray, box, size=(32, 48)):
        x, y, w, h = box
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(gray.shape[1], x + w)
        y2 = min(gray.shape[0], y + h)
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        crop = cv2.equalizeHist(crop)
        vec = crop.astype(np.float32).ravel() / 255.0
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec

    @staticmethod
    def _appearance_feature(frame, box):
        x, y, w, h = box
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        top = y1 + int((y2 - y1) * 0.10)
        bottom = y1 + int((y2 - y1) * 0.88)
        crop = frame[top:bottom, x1:x2]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (64, 96), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).astype(np.float32)
        hist = hist.ravel()
        hist_norm = np.linalg.norm(hist)
        if hist_norm > 1e-8:
            hist /= hist_norm
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (16, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
        small_norm = np.linalg.norm(small)
        if small_norm > 1e-8:
            small /= small_norm
        feat = np.concatenate([hist, small]).astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 1e-8:
            feat /= norm
        return feat

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        union = max(1, aw * ah + bw * bh - inter)
        return inter / union

    @staticmethod
    def _prepare_gray(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def _detect_faces(self, gray, person_box):
        px, py, pw, ph = person_box
        x1 = max(0, px)
        y1 = max(0, py)
        x2 = min(gray.shape[1], px + pw)
        y2 = min(gray.shape[0], py + max(1, int(ph * 0.68)))
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[1] < 70 or roi.shape[0] < 60:
            return None, 'UNKNOWN', 0.0, 0.0
        scale = 1.0
        if roi.shape[1] > 640:
            scale = 640.0 / roi.shape[1]
            roi_small = cv2.resize(roi, (640, max(1, int(roi.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        else:
            roi_small = roi
        candidates = []
        if not self.front.empty():
            try:
                faces = self.front.detectMultiScale(
                    roi_small,
                    scaleFactor=1.03,
                    minNeighbors=4,
                    minSize=(22, 22),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                for fx, fy, fw, fh in faces:
                    if scale != 1.0:
                        fx, fy, fw, fh = [int(v / scale) for v in (fx, fy, fw, fh)]
                    candidates.append(((x1 + fx, y1 + fy, fw, fh), 'FRONT', 0.92))
            except Exception:
                pass
        if not self.profile.empty():
            try:
                faces = self.profile.detectMultiScale(
                    roi_small,
                    scaleFactor=1.03,
                    minNeighbors=3,
                    minSize=(20, 20),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                for fx, fy, fw, fh in faces:
                    if scale != 1.0:
                        fx, fy, fw, fh = [int(v / scale) for v in (fx, fy, fw, fh)]
                    candidates.append(((x1 + fx, y1 + fy, fw, fh), 'LEFT', 0.86))
            except Exception:
                pass
            try:
                flipped = cv2.flip(roi_small, 1)
                faces = self.profile.detectMultiScale(
                    flipped,
                    scaleFactor=1.03,
                    minNeighbors=3,
                    minSize=(20, 20),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                for fx, fy, fw, fh in faces:
                    if scale != 1.0:
                        fx, fy, fw, fh = [int(v / scale) for v in (fx, fy, fw, fh)]
                    fx = flipped.shape[1] - fx - fw
                    candidates.append(((x1 + fx, y1 + fy, fw, fh), 'RIGHT', 0.86))
            except Exception:
                pass
        if not candidates:
            return None, 'UNKNOWN', 0.0, 0.0
        front_candidates = [c for c in candidates if c[1] == 'FRONT']
        profile_candidates = [c for c in candidates if c[1] in ('LEFT', 'RIGHT')]
        if profile_candidates and front_candidates:
            best_profile = max(profile_candidates, key=lambda item: item[0][2] * item[0][3])
            best_front = max(front_candidates, key=lambda item: item[0][2] * item[0][3])
            profile_area = best_profile[0][2] * best_profile[0][3]
            front_area = best_front[0][2] * best_front[0][3]
            if profile_area >= 0.58 * front_area:
                face, side, conf = best_profile
            else:
                face, side, conf = best_front
        else:
            candidates.sort(key=lambda item: item[0][2] * item[0][3], reverse=True)
            face, side, conf = candidates[0]
        fx, fy, fw, fh = self._clamp_box(face, gray.shape[1], gray.shape[0])
        area_ratio = (fw * fh) / max(1.0, pw * ph)
        if fw < 22 or fh < 22 or area_ratio < 0.0025 or area_ratio > 0.30:
            return None, 'UNKNOWN', 0.0, 0.0
        vertical_ratio = ((fy + 0.5 * fh) - py) / max(1.0, float(ph))
        if vertical_ratio > 0.70:
            return None, 'UNKNOWN', 0.0, 0.0
        yaw_map = {'LEFT': -62.0, 'RIGHT': 62.0, 'FRONT': 0.0}
        yaw_deg = yaw_map.get(side, 0.0)
        return (fx, fy, fw, fh), side, conf, yaw_deg

    @staticmethod
    def _mouth_roi(gray, face):
        x, y, w, h = face
        x1 = max(0, int(x + 0.16 * w))
        x2 = min(gray.shape[1], int(x + 0.84 * w))
        y1 = max(0, int(y + 0.55 * h))
        y2 = min(gray.shape[0], int(y + 0.97 * h))
        roi = gray[y1:y2, x1:x2]
        if roi.size < 120:
            return None
        return cv2.resize(roi, (64, 32), interpolation=cv2.INTER_AREA)

    def _mouth_features(self, gray, face):
        roi = self._mouth_roi(gray, face)
        if roi is None:
            return 0.0, 0.0
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        darkness = 1.0 - float(np.mean(roi)) / 255.0
        edges = cv2.Canny(roi, 35, 95)
        edge_density = float(np.mean(edges > 0))
        opening = float(np.clip(0.24 * darkness + 1.45 * edge_density, 0.0, 1.0))
        motion = 0.0
        if self.previous_gray is not None:
            previous_roi = self._mouth_roi(self.previous_gray, face)
            if previous_roi is not None:
                previous_roi = cv2.GaussianBlur(previous_roi, (3, 3), 0)
                motion = float(np.mean(cv2.absdiff(roi, previous_roi)) / 255.0)
        return opening, motion

    def _person_box_ok(self, box, frame_width, frame_height, confidence):
        x, y, w, h = box
        if confidence < 0.55:
            return False
        if w < 28 or h < 50:
            return False
        if x + w <= 1 or y + h <= 1:
            return False
        ratio = h / max(1.0, float(w))
        if ratio < 0.55 or ratio > 4.8:
            return False
        area = (w * h) / max(1.0, frame_width * frame_height)
        if area < 0.003 or area > 0.75:
            return False
        return True

    def _motion_in_region(self, current, previous, box, y_start, y_end):
        if previous is None:
            return 0.0
        x, y, w, h = box
        x1 = max(0, x)
        y1 = max(0, y + int(h * y_start))
        x2 = min(current.shape[1], x + w)
        y2 = min(current.shape[0], y + int(h * y_end))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        a = cv2.cvtColor(current[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        px1 = max(0, min(previous.shape[1] - 1, x1))
        py1 = max(0, min(previous.shape[0] - 1, y1))
        px2 = max(px1 + 1, min(previous.shape[1], x2))
        py2 = max(py1 + 1, min(previous.shape[0], y2))
        b = cv2.cvtColor(previous[py1:py2, px1:px2], cv2.COLOR_BGR2GRAY)
        if a.size == 0 or b.size == 0:
            return 0.0
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        diff = cv2.GaussianBlur(cv2.absdiff(a, b), (3, 3), 0)
        return float(np.clip((float(np.mean(diff)) / 255.0) * 4.5, 0.0, 1.0))

    @staticmethod
    def _nearest_face_anchor(faces, person_box, frame_width, frame_height):
        """Find the face most likely belonging to a person detection.

        YOLO can occasionally return a badly shifted person box (for example,
        a chair/background region next to the student).  YuNet sees the face
        independently, so use the nearest face as a hard geometric anchor.
        """
        if not faces:
            return None
        px, py, pw, ph = [float(v) for v in person_box]
        pcx = px + 0.5 * pw
        pcy = py + 0.22 * ph
        best = None
        best_score = 1e9
        for face in faces:
            try:
                fx, fy, fw, fh = [float(v) for v in face[:4]]
            except Exception:
                continue
            if fw < 20 or fh < 20:
                continue
            fcx = fx + 0.5 * fw
            fcy = fy + 0.5 * fh
            d = math.hypot(fcx - pcx, fcy - pcy)
            # Normalize by the larger expected person dimension.
            gate = max(90.0, 1.35 * max(pw, ph), 0.22 * max(frame_width, frame_height))
            if d > gate:
                continue
            # Prefer a face near the expected head position and with decent size.
            vertical_penalty = abs((fcy - py) / max(1.0, ph) - 0.22)
            score = d + 120.0 * vertical_penalty - 0.15 * math.sqrt(fw * fh)
            if score < best_score:
                best_score = score
                best = face
        return best

    @staticmethod
    def _recenter_person_box_on_face(person_box, face, width, height, strength=0.92):
        """Recenter a person box so its upper body is actually attached to face."""
        if face is None:
            return person_box
        try:
            x, y, w, h = [float(v) for v in person_box]
            fx, fy, fw, fh = [float(v) for v in face[:4]]
            fcx = fx + 0.5 * fw
            # The face should sit roughly 20% from the top of the person box.
            desired_x = fcx - 0.50 * w
            desired_y = (fy + 0.5 * fh) - 0.20 * h
            new_x = x + float(strength) * (desired_x - x)
            new_y = y + float(strength) * (desired_y - y)
            return SmartVision._clamp_box((
                int(round(new_x)), int(round(new_y)),
                int(round(max(w, fw * 2.45))),
                int(round(max(h, fh * 3.25))),
            ), width, height)
        except Exception:
            return person_box

    @staticmethod
    def _object_name(cls_id):
        return {63: 'LAPTOP', 65: 'REMOTE', 67: 'PHONE', 73: 'BOOK'}.get(int(cls_id), 'OBJECT')

    def detect(self, frame):
        start = time.perf_counter()
        self.frame_index += 1
        height, width = frame.shape[:2]
        max_width = 960
        scale = 1.0
        if width > max_width:
            scale = max_width / float(width)
            small = cv2.resize(frame, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        else:
            small = frame

        results = None
        if self.model is not None:
            try:
                results = self.model.predict(
                    source=small,
                    classes=[0, 63, 65, 67, 73],
                    conf=0.35,
                    iou=0.48,
                    imgsz=640,
                    max_det=40,
                    device='cpu',
                    verbose=False,
                )
            except Exception as exc:
                self.last_model_error = f'YOLO inference failed: {exc}; using YuNet Face Fallback.'
                self.model = None
                self.model_source = 'YuNet Face Fallback'
        gray = self._prepare_gray(frame)
        persons = []
        objects = []
        if results:
            boxes = getattr(results[0], 'boxes', None)
            if boxes is not None:
                for i in range(len(boxes)):
                    try:
                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        coords = boxes.xyxy[i].cpu().numpy().tolist()
                    except Exception:
                        continue
                    if scale != 1.0:
                        inv = 1.0 / scale
                        coords = [float(v) * inv for v in coords]
                    x1, y1, x2, y2 = coords
                    box = self._clamp_box((int(x1), int(y1), int(x2 - x1), int(y2 - y1)), width, height)
                    if cls_id == 0:
                        if conf >= 0.42 and self._person_box_ok(box, width, height, conf):
                            persons.append((box, conf))
                    elif cls_id in (63, 65, 67, 73) and conf >= 0.30:
                        if box[2] >= 18 and box[3] >= 12:
                            objects.append((box, cls_id, conf))
        yunet_faces = self._yunet_faces(frame)

        # Hard geometric face anchor: this fixes badly shifted YOLO person
        # boxes and prevents the lock box from sitting on a chair/background.
        if persons and yunet_faces:
            anchored_persons = []
            for person_box, person_conf in persons:
                anchor = self._nearest_face_anchor(yunet_faces, person_box, width, height)
                if anchor is not None:
                    person_box = self._recenter_person_box_on_face(
                        person_box, anchor, width, height, strength=0.92
                    )
                anchored_persons.append((person_box, person_conf))
            persons = anchored_persons

        # No YOLO model (or YOLO failed): use YuNet face boxes as person tracks.
        # This keeps Face ID + tracking functional without requiring a large
        # YOLO weights file on the development machine.
        if not persons and yunet_faces:
            for face in yunet_faces[:40]:
                try:
                    fx, fy, fw, fh = [float(v) for v in face[:4]]
                except Exception:
                    continue
                # Expand the face region into a stable upper-body tracking box.
                x = int(fx - 0.65 * fw)
                y = int(fy - 0.45 * fh)
                w = int(fw * 2.30)
                h = int(fh * 3.10)
                box = self._clamp_box((x, y, w, h), width, height)
                conf = float(face[14]) if len(face) >= 15 else 0.85
                conf = max(0.62, min(0.99, conf))
                if self._person_box_ok(box, width, height, conf):
                    persons.append((box, conf))

        detections = []
        for person_box, person_conf in persons:
            appearance = self._appearance_feature(frame, person_box)
            face, side, face_conf, yaw_deg = self._detect_faces(gray, person_box)
            yunet_face = self._select_yunet_face(yunet_faces, person_box)
            if face is None and yunet_face is not None:
                ux, uy, uw, uh = [int(float(v)) for v in yunet_face[:4]]
                face = self._clamp_box((ux, uy, uw, uh), width, height)
                side = 'UNKNOWN'
                face_conf = float(yunet_face[14]) if len(yunet_face) >= 15 else 0.65
                yaw_deg = 0.0
            head_dir = 0.0
            head_quality = 0.0
            mouth_score = 0.0
            mouth_motion = 0.0
            mouth_activity = 0.0
            face_feature = None
            face_y_ratio = 0.0
            if face is not None:
                fx, fy, fw, fh = face
                face_feature = self._sface_feature(frame, yunet_face) if yunet_face is not None else None
                if face_feature is None:
                    face_feature = self._gray_feature(gray, face)
                mouth_score, mouth_motion = self._mouth_features(gray, face)
                mouth_activity = float(np.clip(0.55 * mouth_score + 5.5 * mouth_motion, 0.0, 1.0))
                face_y_ratio = ((fy + 0.5 * fh) - person_box[1]) / max(1.0, float(person_box[3]))
                head_quality = max(0.0, min(1.0, math.sqrt(fw * fh) / 74.0)) * face_conf
                if side == 'LEFT':
                    head_dir = -1.0
                elif side == 'RIGHT':
                    head_dir = 1.0
            x, y, w, h = person_box
            attached_objects = []
            for obj_box, cls_id, obj_conf in objects:
                ox, oy, ow, oh = obj_box
                ocx = ox + ow * 0.5
                ocy = oy + oh * 0.5
                inside = x <= ocx <= x + w and y <= ocy <= y + h
                overlap = self._iou(person_box, obj_box)
                if inside or overlap >= 0.08:
                    attached_objects.append((self._object_name(cls_id), obj_conf))
            lower_motion = self._motion_in_region(frame, self.previous_frame, person_box, 0.42, 0.98)
            hand_motion = self._motion_in_region(frame, self.previous_frame, person_box, 0.48, 0.83)
            task_activity = float(np.clip(0.48 * lower_motion + 0.52 * hand_motion, 0.0, 1.0))
            if attached_objects:
                task_activity = max(task_activity, min(1.0, 0.64 + 0.18 * max(v for _, v in attached_objects)))
            detections.append(Detection(
                box=person_box,
                person_score=person_conf,
                face=face,
                face_feature=face_feature,
                appearance_feature=appearance,
                face_side=side,
                head_dir=head_dir,
                head_yaw_deg=float(yaw_deg),
                head_quality=float(head_quality),
                mouth_score=float(mouth_score),
                mouth_motion=float(mouth_motion),
                mouth_activity=float(mouth_activity),
                face_y_ratio=float(face_y_ratio),
                object_types=tuple(name for name, _ in attached_objects),
                object_scores=tuple(float(score) for _, score in attached_objects),
                lower_motion=float(lower_motion),
                hand_motion=float(hand_motion),
                task_activity=float(task_activity),
            ))
        detections.sort(key=lambda d: d.person_score, reverse=True)
        detections = detections[:40]
        self.last_detections = detections
        self.previous_gray = gray.copy()
        self.previous_frame = frame.copy()
        self.inference_count += 1
        self.last_inference_ms = (time.perf_counter() - start) * 1000.0
        return [d.__dict__ for d in detections], objects


class BehaviorEngine:
    """
    God Eyes classroom behavior engine.

    Rules:
    1) Head turn LEFT/RIGHT:
       - abs(yaw) > 50 degrees
       - continuously for 5 seconds
       - emit one event per turn episode.

    2) Locked student + face suddenly unavailable:
       - keep the existing person box/identity alive;
       - inspect the tracked box motion, not only detector-side face motion;
       - any meaningful body-box movement keeps the student in TRACK_HOLD;
       - if the tracked box is effectively stationary for 5 continuous seconds,
         emit one OB_SLEEP observation for that locked student.

    3) When the face returns, IdentityLock re-verifies its embedding against the
       complete enrolled roster. Only the same locked student is allowed to
       resume; a different student's embedding can never steal this track.

    These are visual observation signals and should be presented to the teacher
    as signals requiring confirmation, not as definitive conclusions.
    """

    YAW_THRESHOLD_DEG = 50.0
    YAW_CONFIRM_SECONDS = 5.0

    NO_FACE_STILL_SLEEP_SECONDS = 5.0
    FACE_GAP_GRACE_SECONDS = 0.40

    # A short-window box-center displacement, normalized by tracked box height.
    # This is intentionally small: even light genuine movement should prevent
    # the sleep timer from firing, while sub-pixel/box-jitter noise is ignored.
    BOX_MOVE_WINDOW_SECONDS = 0.80
    BOX_MOVE_THRESHOLD = 0.018
    BOX_SIZE_CHANGE_THRESHOLD = 0.025

    EVENT_COOLDOWN = 8.0

    def __init__(self, board_side='RIGHT'):
        self.board_side = board_side
        self.state = {}

    def set_board_side(self, board_side):
        self.board_side = board_side

    def _state(self, sid):
        if sid not in self.state:
            self.state[sid] = {
                'turn_side': 0,
                'turn_start': None,
                'turn_latched': False,
                'last_yaw': 0.0,
                'last_face_time': 0.0,
                'face_missing_start': None,
                'sleep_latched': False,
                'last_event': {},
                'last_seen': 0.0,
            }
        return self.state[sid]

    def _cooldown_ok(self, state, key, now, seconds=None):
        seconds = self.EVENT_COOLDOWN if seconds is None else float(seconds)
        last = float(state['last_event'].get(key, 0.0))
        return (now - last) >= seconds

    @staticmethod
    def _event(key, confidence, details):
        return key, float(confidence), str(details)

    @staticmethod
    def _box_motion_in_window(track, now, window_seconds):
        """Measure recent tracked-box movement, normalized by box height.

        The tracker keeps a filtered center history even during detector misses.
        We look back a short time window so a little real movement prevents a
        false sleep signal, while tiny per-frame jitter does not reset the timer.
        """
        history = list(getattr(track, 'center_history', []) or [])
        if len(history) < 2:
            return 0.0, 0.0

        box = getattr(track, 'box', (0.0, 0.0, 100.0, 100.0))
        height = max(60.0, float(box[3]))
        current_center = history[-1]
        current_index = len(history) - 1

        # center_history does not store timestamps, so use a conservative frame
        # rate estimate from the AI loop.  At 6-12 FPS this window corresponds to
        # roughly 5-10 recent samples.
        samples_back = max(2, int(round(float(window_seconds) * 10.0)))
        start_index = max(0, current_index - samples_back)
        reference = history[start_index]
        try:
            displacement = math.hypot(
                float(current_center[0]) - float(reference[0]),
                float(current_center[1]) - float(reference[1]),
            ) / height
        except Exception:
            displacement = 0.0

        # Also consider box size change. A student writing/adjusting posture can
        # change the tracked box even when the center hardly moves.
        size_change = 0.0
        try:
            # Height history is aligned with center history in PersonTrack.
            heights = list(getattr(track, 'height_history', []) or [])
            if heights:
                ref_h_index = max(0, len(heights) - 1 - (current_index - start_index))
                ref_h = max(1.0, float(heights[ref_h_index]))
                cur_h = max(1.0, float(heights[-1]))
                size_change = abs(cur_h - ref_h) / max(ref_h, cur_h)
        except Exception:
            size_change = 0.0

        return float(displacement), float(size_change)

    def evaluate(self, sid, track, now=None):
        now = time.time() if now is None else float(now)
        state = self._state(sid)
        state['last_seen'] = now
        events = []

        face_valid = (
            getattr(track, 'face', None) is not None
            and float(getattr(track, 'head_quality', 0.0)) >= 0.12
        )

        yaw = float(getattr(track, 'head_yaw_deg', 0.0))
        head_quality = float(getattr(track, 'head_quality', 0.0))
        person_score = float(getattr(track, 'person_score', 0.0))

        if face_valid:
            state['last_yaw'] = yaw
            state['last_face_time'] = now

        # ---------------------------------------------------------
        # 1) HEAD TURN > 50°, continuously for 5 seconds
        # ---------------------------------------------------------
        turned = bool(face_valid and abs(yaw) > self.YAW_THRESHOLD_DEG)
        direction = -1 if yaw < 0 else 1 if yaw > 0 else 0

        if turned:
            if state['turn_side'] != direction:
                state['turn_side'] = direction
                state['turn_start'] = now
                state['turn_latched'] = False

            turn_start = state.get('turn_start')
            elapsed = now - float(turn_start) if turn_start is not None else 0.0

            if elapsed >= self.YAW_CONFIRM_SECONDS and not state['turn_latched']:
                key = 'HEAD_TURN_LEFT' if direction < 0 else 'HEAD_TURN_RIGHT'
                side = 'left' if direction < 0 else 'right'
                confidence = min(
                    0.97,
                    0.60
                    + 0.22 * min(1.0, head_quality)
                    + 0.15 * min(1.0, person_score),
                )
                events.append(self._event(
                    key,
                    confidence,
                    (
                        f'Đầu quay {side} quá '
                        f'{self.YAW_THRESHOLD_DEG:.0f}° liên tục '
                        f'{self.YAW_CONFIRM_SECONDS:.0f} giây.'
                    ),
                ))
                state['last_event'][key] = now
                state['turn_latched'] = True
        elif (
            not face_valid
            and state['turn_side'] != 0
            and (now - float(state.get('last_face_time', 0.0))) <= self.FACE_GAP_GRACE_SECONDS
        ):
            pass
        else:
            state['turn_side'] = 0
            state['turn_start'] = None
            state['turn_latched'] = False

        # ---------------------------------------------------------
        # 2) LOCKED TRACK + FACE LOST
        # ---------------------------------------------------------
        if face_valid:
            # Face returned. The IdentityLock layer will re-verify its embedding
            # against the complete roster before allowing the identity to resume.
            state['face_missing_start'] = None
            state['sleep_latched'] = False
        else:
            displacement, size_change = self._box_motion_in_window(
                track, now, self.BOX_MOVE_WINDOW_SECONDS
            )
            moving = (
                displacement >= self.BOX_MOVE_THRESHOLD
                or size_change >= self.BOX_SIZE_CHANGE_THRESHOLD
            )

            if moving:
                # The same lock box is still changing, even if the face is gone.
                # Do not call it sleep; keep the lock and keep following.
                state['face_missing_start'] = None
                state['sleep_latched'] = False
            else:
                if state['face_missing_start'] is None:
                    state['face_missing_start'] = now

                missing_elapsed = now - float(state['face_missing_start'])
                if (
                    missing_elapsed >= self.NO_FACE_STILL_SLEEP_SECONDS
                    and not state['sleep_latched']
                    and self._cooldown_ok(state, 'OB_SLEEP', now)
                ):
                    confidence = min(
                        0.92,
                        0.66
                        + 0.10 * min(1.0, person_score)
                        + 0.10 * (1.0 - min(1.0, displacement / max(self.BOX_MOVE_THRESHOLD, 1e-6))),
                    )
                    events.append(self._event(
                        'OB_SLEEP',
                        confidence,
                        (
                            'Khuôn mặt tạm thời không còn nhìn thấy; '
                            f'khung người gần như đứng yên liên tục {self.NO_FACE_STILL_SLEEP_SECONDS:.0f} giây. '
                            'Đây là tín hiệu nghi ngờ ngủ/gục và cần giáo viên xác nhận.'
                        ),
                    ))
                    state['last_event']['OB_SLEEP'] = now
                    state['sleep_latched'] = True

        return events

    def reset(self):
        self.state.clear()


class AIWorker(QThread):
    tracks_ready = Signal(object)
    status = Signal(str)
    event_ready = Signal(str, str, float, str, object)

    def __init__(self, camera, model_dir, board_side='RIGHT'):
        super().__init__()
        self.camera = camera
        self.model_dir = Path(model_dir)
        self.board_side = board_side
        self.vision = None
        self.tracker = PersonTracker()
        self.identity = IdentityLock()
        self.behavior = BehaviorEngine(board_side)
        self.latest_frame = None
        self._frame_lock = threading.Lock()
        self._config_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self.active = True
        self.mode = 'scan'
        self.session_id = None
        self.target_fps = 10.0
        self.next_run = 0.0
        self.last_status = ''
        self.last_error_time = 0.0
        self.last_emit_time = 0.0
        self.total_frames_received = 0
        self.total_inferences = 0
        self.last_frame_time = 0.0
        self.camera.frame_ready.connect(self._receive_frame)

    def _receive_frame(self, frame):
        if frame is None:
            return
        try:
            if frame.size == 0:
                return
            copied = frame.copy()
        except Exception:
            return
        with self._frame_lock:
            self.latest_frame = copied
            self.total_frames_received += 1
            self.last_frame_time = time.time()

    def set_server_roster(self, roster):
        with self._state_lock:
            self.identity.set_server_roster(roster)

    def configure(self, session_id, mode, fps, board_side):
        with self._config_lock:
            self.session_id = session_id
            self.mode = str(mode)
            self.target_fps = max(6.0, min(12.0, float(fps)))
            self.board_side = board_side
            self.behavior.set_board_side(board_side)
            self.active = True
            self.next_run = 0.0
            if self.mode == 'scan':
                self.status.emit('AI SCAN • finding stable person tracks…')
            else:
                self.status.emit('AI MONITOR • person detection active')

    def set_active(self, active):
        self.active = bool(active)
        if not active:
            self.status.emit('PAUSED • AI events disabled')
        else:
            self.status.emit('AI RESUMED • processing newest camera frame')

    def reset_identity(self):
        with self._state_lock:
            self.identity.reset()
            self.tracker.reset()
            self.behavior.reset()
        self.status.emit('IDENTITY RESET • scanning again')

    def lock_ids(self):
        with self._frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        frame_size = None
        if frame is not None:
            frame_size = (frame.shape[1], frame.shape[0])
        with self._state_lock:
            mapping = self.identity.lock(self.tracker.tracks.values(), frame_size)
        self.tracker.set_protected_track_ids(mapping.keys() if mapping else [])
        if mapping:
            self.status.emit(f'IDENTITIES LOCKED • {len(mapping)} students • SMOOTH FOLLOW ON')
        else:
            self.status.emit('IDENTITY LOCK FAILED • no stable students')
        return mapping

    def mapping(self):
        with self._state_lock:
            return self.identity.mapping()

    def stop(self):
        self._stop_event.set()
        self.active = False

    def _emit_status(self, text, min_interval=0.8):
        now = time.time()
        if text != self.last_status or now - self.last_emit_time >= min_interval:
            self.last_status = text
            self.last_emit_time = now
            self.status.emit(text)

    def _status_error(self, label, exc):
        message = str(exc).strip().replace('\n', ' ')
        if not message:
            message = exc.__class__.__name__
        if len(message) > 240:
            message = message[:240] + '…'
        self.status.emit(f'{label} • {message}')

    @staticmethod
    def _draw_evidence(frame, track, sid, event_type=None, confidence=None):
        image = frame.copy()
        x, y, w, h = [int(v) for v in track.box]
        pen = (30, 40, 235)
        cv2.rectangle(image, (x, y), (x + w, y + h), pen, 3)
        label = str(sid)
        if event_type:
            label = f'{sid} • {event_type.replace("_", " ").upper()}'
        if confidence is not None:
            label = f'{label} • {confidence * 100:.0f}%'
        top = max(32, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.50
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        right = min(image.shape[1], x + tw + 16)
        cv2.rectangle(image, (x, top - 30), (right, top), pen, -1)
        cv2.putText(image, label, (x + 7, top - 9), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        yaw = float(getattr(track, 'head_yaw_deg', 0.0))
        mouth = float(getattr(track, 'mouth_activity', 0.0))
        meta = f'Yaw {yaw:+.0f} deg  |  Mouth activity {mouth * 100:.0f}%'
        cv2.rectangle(image, (x, min(image.shape[0] - 28, y + h)), (min(image.shape[1], x + 245), min(image.shape[0], y + h + 24)), (20, 35, 55), -1)
        cv2.putText(image, meta, (x + 6, min(image.shape[0] - 8, y + h + 17)), font, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def _payload_for_track(self, track, sid, score):
        return {
            'track_id': track.track_id,
            'student': sid,
            'student_id': int(self.identity.student_id_for_label(sid)) if self.identity.server_roster else 0,
            'identity_source': str(self.identity.student_profiles.get(sid, {}).get('identity_source', 'UNKNOWN')) if sid not in ('IDENTITY UNCERTAIN',) else 'UNKNOWN',
            'missed': track.missed,
            'score': float(max(0.0, min(1.0, score))),
            'box': tuple(int(v) for v in track.box),
            'confirmed': bool(track.confirmed),
            'person_confidence': float(track.person_score),
            'face_side': track.face_side,
            'face_quality': float(track.head_quality),
            'head_yaw_deg': float(getattr(track, 'head_yaw_deg', 0.0)),
            'mouth_activity': float(getattr(track, 'mouth_activity', 0.0)),
            'mouth_motion': float(getattr(track, 'mouth_motion', 0.0)),
            'object_types': list(getattr(track, 'object_types', ()) or ()),
            'task_activity': float(getattr(track, 'task_activity', 0.0)),
            'lower_motion': float(getattr(track, 'lower_motion', 0.0)),
            'hand_motion': float(getattr(track, 'hand_motion', 0.0)),
        }

    def _scan_face_detections(self, frame):
        """Face-only detections for the attendance/Face ID scan stage.

        The scan stage should lock identities from the head/face area only.
        Full-person boxes are still used in monitor mode because posture and
        hand/body motion are needed there.
        """
        if self.vision is None or self.vision.face_detector is None:
            return []
        faces=self.vision._yunet_faces(frame)
        gray=self.vision._prepare_gray(frame)
        detections=[]
        h,w=frame.shape[:2]
        for face in faces[:40]:
            try:
                fx,fy,fw,fh=[float(v) for v in face[:4]]
            except Exception:
                continue
            box=self.vision._clamp_box((int(fx),int(fy),int(fw),int(fh)),w,h)
            x,y,bw,bh=box
            if bw < 30 or bh < 30:
                continue
            conf=float(face[14]) if len(face) >= 15 else 0.85
            if conf < 0.55:
                continue
            feature=self.vision._sface_feature(frame, face)
            if feature is None:
                feature=self.vision._gray_feature(gray, box)
            mouth_score,mouth_motion=self.vision._mouth_features(gray, box)
            mouth_activity=float(np.clip(0.55*mouth_score + 5.5*mouth_motion,0.0,1.0))
            quality=max(0.0,min(1.0,math.sqrt(bw*bh)/74.0))*conf
            detections.append(Detection(
                box=box, person_score=max(0.70,min(0.99,conf)), face=box,
                face_feature=feature, appearance_feature=self.vision._appearance_feature(frame, box),
                face_side='UNKNOWN', head_dir=0.0, head_yaw_deg=0.0,
                head_quality=float(quality), mouth_score=float(mouth_score),
                mouth_motion=float(mouth_motion), mouth_activity=float(mouth_activity),
                face_y_ratio=0.5, object_types=(), object_scores=(),
                lower_motion=0.0, hand_motion=0.0, task_activity=0.0
            ))
        return detections[:40]

    def _process_once(self, frame):
        if self.vision is None:
            self._emit_status('LOADING YOLO PERSON DETECTOR…', 1.0)
            try:
                self.vision = SmartVision(self.model_dir)
            except Exception as exc:
                self.last_error_time = time.time()
                self._status_error('AI MODEL ERROR', exc)
                self.tracks_ready.emit([])
                return
            face_mode = 'SFACE READY' if self.vision.face_recognizer is not None else 'SFACE MODEL MISSING'
            self._emit_status(f'AI READY • {Path(self.vision.model_source).name} • {face_mode} • LOCKED TRACK • SMOOTH FOLLOW • HEAD TURN >50°/5s • NO-FACE+STILL 5s • SESSION-LONG ID LOCK • EMBEDDING GUARD', 0.5)

        if self.mode == 'scan':
            detections = self._scan_face_detections(frame)
        else:
            raw_detections, _ = self.vision.detect(frame)
            # SmartVision returns serializable dicts; PersonTracker operates on
            # Detection instances.  Converting here prevents the recurring
            # 'dict has no attribute box' AI PROCESS ERROR loop.
            detections = [
                Detection(**item) if isinstance(item, dict) else item
                for item in (raw_detections or [])
            ]
        with self._state_lock:
            self.tracker.set_protected_track_ids(self.identity.mapping().keys())
            track_objects = self.tracker.update(detections)
            # IdentityLock needs the current tracker state before resolving
            # identities, so a lost old track can be handed off safely to a
            # newly created track of the same student.
            self.identity.begin_frame(track_objects)
            # Once a student is locked, make a final face-anchor correction on
            # that track. This keeps the visible box attached to the actual
            # person even when the detector's body box jitters.
            if self.mode == 'monitor' and getattr(self.vision, 'face_detector', None) is not None:
                face_anchors = self.vision._yunet_faces(frame)
                if face_anchors:
                    for track in track_objects:
                        if track.track_id not in self.identity.mapping():
                            continue
                        anchor = self.vision._nearest_face_anchor(
                            face_anchors, track.box, frame.shape[1], frame.shape[0]
                        )
                        if anchor is None:
                            continue
                        corrected = self.vision._recenter_person_box_on_face(
                            track.box, anchor, frame.shape[1], frame.shape[0], strength=0.22
                        )
                        track.box = tuple(
                            float(a) * 0.58 + float(b) * 0.42
                            for a, b in zip(track.box, corrected)
                        )
        frame_size = (frame.shape[1], frame.shape[0])
        payload = []
        active_student_ids = set()
        now = time.time()

        for track in track_objects:
            if track.missed > self.tracker.max_missed:
                continue
            with self._state_lock:
                sid, identity_score = self.identity.resolve(track, frame_size)
            if track.missed > 0:
                display_score = identity_score * 0.88
            else:
                display_score = identity_score
            payload.append(self._payload_for_track(track, sid, display_score))
            if track.visible and sid not in ('IDENTITY UNCERTAIN',) and track.confirmed:
                active_student_ids.add(sid)
            is_server_identity = bool(self.identity.server_roster and self.identity.student_id_for_label(sid) > 0)
            is_local_identity = sid.startswith('HS ')

            # IMPORTANT: behavior evaluation must continue while the person track
            # is temporarily missed. IdentityLock intentionally keeps the student
            # identity during short face-loss periods, which is required for the
            # 5-second OB_SLEEP rule. Only the identity/track must be valid; the
            # face itself does not need to remain visible.
            if self.mode == 'monitor' and track.confirmed and sid not in ('IDENTITY UNCERTAIN',) and (is_server_identity or is_local_identity):
                with self._state_lock:
                    behavior_events = self.behavior.evaluate(sid, track, now)
                for event_type, confidence, assessment in behavior_events:
                    evidence = self._draw_evidence(frame, track, sid, event_type, confidence)
                    self.event_ready.emit(sid, event_type, float(confidence), assessment, evidence)

        self.total_inferences += 1
        visible_tracks = sum(1 for item in payload if item['missed'] == 0 and item['confirmed'])
        if self.vision.last_inference_ms > 0:
            self._emit_status(
                f'AI ONLINE • {visible_tracks}/40 PERSON(S) • {self.vision.last_inference_ms:.0f}ms',
                1.5,
            )
        else:
            self._emit_status(f'AI ONLINE • {visible_tracks}/40 PERSON(S)', 1.5)
        self.tracks_ready.emit(payload)

    def run(self):
        self._stop_event.clear()
        self._emit_status('AI THREAD STARTED', 0.0)
        try:
            while not self._stop_event.is_set():
                if not self.active:
                    time.sleep(0.05)
                    continue
                interval = 1.0 / max(1.0, self.target_fps)
                now = time.time()
                if now < self.next_run:
                    time.sleep(min(0.03, max(0.0, self.next_run - now)))
                    continue
                self.next_run = now + interval
                with self._frame_lock:
                    frame = None if self.latest_frame is None else self.latest_frame.copy()
                    last_frame_time = self.last_frame_time
                if frame is None:
                    self._emit_status('AI WAITING • no camera frame received', 1.5)
                    time.sleep(0.05)
                    continue
                if last_frame_time and time.time() - last_frame_time > 2.5:
                    self._emit_status('AI WAITING • camera frame is stale', 1.0)
                try:
                    self._process_once(frame)
                except Exception as exc:
                    self._status_error('AI PROCESS ERROR', exc)
                    traceback.print_exc()
                    time.sleep(0.35)
        finally:
            self._emit_status('AI STOPPED', 0.0)
