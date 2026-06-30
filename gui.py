#!/usr/bin/env python3
"""
ASCEND Ground Station GUI — Team Anveshak
==========================================
Colour palette (from logo):
  Orange accent  : #E8621A
  Dark bg        : #1A1A1A
  Panel bg       : #212121
  Card bg        : #2B2B2B
  Border         : #333333
  Text primary   : #F0F0F0
  Text secondary : #9E9E9E
  Green ok       : #00C853
  Amber warn     : #FF8F00
  Red crit       : #D32F2F
  Blue info      : #1E88E5

Requires:
    pip install PyQt6
    ROS2 + mavros_msgs installed in the environment.

All data displayed is live from ROS2 topics only.
No simulation, no fake data.
"""

import sys
import math
import time
import json
import base64
import threading
from datetime import datetime
from std_msgs.msg import Int32, Bool
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QFrame, QProgressBar, QPushButton, QScrollArea,
)
from PyQt6.QtCore import Qt, QTimer, QThread, QRectF, QPointF
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath, QPixmap,
)


# ─────────────────────────── COLOUR TOKENS ───────────────────────────
C_BG     = QColor("#1A1A1A")
C_PANEL  = QColor("#212121")
C_CARD   = QColor("#2B2B2B")
C_BORDER = QColor("#333333")
C_ORANGE = QColor("#E8621A")
C_TEXT   = QColor("#F0F0F0")
C_MUTED  = QColor("#9E9E9E")
C_GREEN  = QColor("#00C853")
C_AMBER  = QColor("#FF8F00")
C_RED    = QColor("#D32F2F")
C_BLUE   = QColor("#1E88E5")


# ─────────────────────────── MAVROS MODES ────────────────────────────
MAVROS_MODE_COLORS = {
    "MANUAL":       C_MUTED,
    "STABILIZED":   C_MUTED,
    "ALTCTL":       C_BLUE,
    "POSCTL":       C_BLUE,
    "OFFBOARD":     C_GREEN,
    "AUTO.TAKEOFF": C_ORANGE,
    "AUTO.LOITER":  C_AMBER,
    "AUTO.RTL":     C_AMBER,
    "AUTO.LAND":    C_RED,
    "AUTO.MISSION": C_GREEN,
    "ACRO":         C_MUTED,
    "RATTITUDE":    C_MUTED,
    "EMERGENCY":    C_RED,
}

def mode_color(mode: str) -> QColor:
    return MAVROS_MODE_COLORS.get(mode, C_MUTED)


FALLBACKS = {
    "Low Battery":   "LAND immediately",
    "Odometry Loss": "LAND using IMU",
    "Comms Lost":    "AUTO.LAND mode",
    "ArUco Lost":    "Hover → re-scan",
    "Align Timeout": "LAND at current pos",
    "Motor Anomaly": "DISARM immediately",
}


# ─────────────────────────── SHARED DATA STORE ───────────────────────
class DroneData:
    """Thread-safe container. All values are raw from ROS2 — no faking."""

    def __init__(self):
        self._lock = threading.Lock()
        # Raw odometry from MAVROS
        self.raw_x = self.raw_y = self.raw_z = 0.0
        # Origin offset (set by operator via "Set Origin" button)
        self.origin_x = self.origin_y = self.origin_z = 0.0
        # Attitude
        self.roll = self.pitch = self.yaw = 0.0
        # Velocity
        self.vx = self.vy = self.vz = self.speed = 0.0
        # Battery — total voltage only
        self.total_v = 0.0
        self.connected_battery = False
        # Field detection
        self.aruco_detected = False
        self.aruco_debug_jpeg = None   # latest raw JPEG bytes from /aruco_debug_image/compress
        # MAVROS state
        self.mode = "—"
        self.armed = False
        self.connected_mavros = False
        # Image transfer state — matches drone_control.py /transfer_status (Int32)
        # 0 = inflight, 1 = land phase / transfer initiated, 2 = transfer complete
        self.transfer_status = 0
        # Feature detection state (from feature_detect_ros2.py)
        self.fd_status = "—"            # "RUNNING" | "DONE" | "ERROR" | "—"
        self.fd_current_image = "—"
        self.fd_progress_current = 0
        self.fd_progress_total = 0
        self.fd_results = []            # list of dicts, see add_feature_result()
        # Log
        self.log_entries = []   # [(time_str, msg, level), ...]

    # Derived position (relative to operator-set origin)
    @property
    def x(self): return self.raw_x - self.origin_x
    @property
    def y(self): return self.raw_y - self.origin_y
    @property
    def z(self): return self.raw_z - self.origin_z

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def set_origin_here(self):
        """Capture current raw position as the new map origin."""
        with self._lock:
            self.origin_x = self.raw_x
            self.origin_y = self.raw_y
            self.origin_z = self.raw_z
            self.log_entries.append((
                '0',
                f"Origin set → raw ({self.origin_x:.2f}, "
                f"{self.origin_y:.2f}, {self.origin_z:.2f})",
                "info",
            ))

    def snapshot(self):
        with self._lock:
            import copy
            return copy.copy(self)

    def add_log(self, msg, level="info"):
        with self._lock:
            self.log_entries.append(('0', msg, level))
            if len(self.log_entries) > 100:
                self.log_entries.pop(0)

    def add_feature_result(self, result: dict):
        """Thread-safe append to fd_results, capped at 200 (same pattern as log)."""
        with self._lock:
            self.fd_results.append(result)
            if len(self.fd_results) > 200:
                self.fd_results.pop(0)


# ─────────────────────── ROS2 WORKER THREAD ──────────────────────────
class Ros2Worker(QThread):
    """Subscribes to MAVROS and ESP32 battery topics. No fallback sim."""

    def __init__(self, data: DroneData):
        super().__init__()
        self.data = data
        self._stop = threading.Event()

    def run(self):
        try:
            import rclpy
            from rclpy.node import Node
            from nav_msgs.msg import Odometry
            from mavros_msgs.msg import State
            from std_msgs.msg import Float32, String
            from sensor_msgs.msg import CompressedImage
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

            rclpy.init()

            class _Inner(Node):
                def __init__(self_, data_ref):
                    super().__init__('ascend_gui')
                    self_.data = data_ref
                    qos = QoSProfile(
                        reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST,
                        depth=10,
                    )
                    self_.create_subscription(
                        Odometry, '/mavros/local_position/odom',
                        self_._cb_odom, qos)
                    self_.create_subscription(
                        State, '/mavros/state',
                        self_._cb_state, qos)
                    self_.create_subscription(
                        Float32, 'esp/total_voltage',
                        self_._cb_total_v, qos)
                    self_.create_subscription(Bool, 'aruco_detected', self_._cb_aruco, qos)
                    self_.create_subscription(
                        CompressedImage, '/aruco_debug_image/compress',
                        self_._cb_aruco_debug_image, qos)

                    # Image transfer status — published by drone_control.py
                    self_.create_subscription(
                        Int32, '/transfer_status',
                        self_._cb_transfer_status, qos)

                    # Feature detection topics — published by feature_detect_ros2.py
                    self_.create_subscription(
                        String, 'feature_detection/progress',
                        self_._cb_fd_progress, qos)
                    self_.create_subscription(
                        String, 'feature_detection/result',
                        self_._cb_fd_result, qos)
                    self_.create_subscription(
                        String, 'feature_detection/status',
                        self_._cb_fd_status, qos)

                def _cb_odom(self_, msg):
                    pos = msg.pose.pose.position
                    q   = msg.pose.pose.orientation
                    vel = msg.twist.twist.linear
                    roll, pitch, yaw = _quat_to_euler(q.x, q.y, q.z, q.w)
                    spd = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    self_.data.update(
                        raw_x=pos.x, raw_y=pos.y, raw_z=pos.z,
                        roll=roll, pitch=pitch, yaw=yaw,
                        vx=vel.x, vy=vel.y, vz=vel.z,
                        speed=spd,
                        connected_mavros=True,
                    )

                def _cb_state(self_, msg):
                    self_.data.update(
                        mode=msg.mode,
                        armed=msg.armed,
                        connected_mavros=msg.connected,
                    )

                def _cb_total_v(self_, msg):
                    self_.data.update(total_v=msg.data, connected_battery=True)

                def _cb_aruco(self_, msg):
                    self_.data.update(aruco_detected=msg.data)

                def _cb_aruco_debug_image(self_, msg):
                    try:
                        self_.data.update(aruco_debug_jpeg=bytes(msg.data))
                    except Exception as e:
                        self_.data.add_log(f"Bad ArUco debug frame: {e}", "warn")

                def _cb_transfer_status(self_, msg):
                    self_.data.update(transfer_status=msg.data)
                    if msg.data == 1:
                        self_.data.add_log("Image transfer started…", "info")
                    elif msg.data == 2:
                        self_.data.add_log("Image transfer complete", "info")

                def _cb_fd_progress(self_, msg):
                    try:
                        d = json.loads(msg.data)
                        self_.data.update(
                            fd_progress_current=int(d.get("current", 0)),
                            fd_progress_total=int(d.get("total", 0)),
                            fd_current_image=str(d.get("image", "—")),
                        )
                    except Exception as e:
                        self_.data.add_log(f"Bad progress message: {e}", "warn")

                def _cb_fd_result(self_, msg):
                    try:
                        d = json.loads(msg.data)
                        self_.data.add_feature_result(d)
                    except Exception as e:
                        self_.data.add_log(f"Bad result message: {e}", "warn")

                def _cb_fd_status(self_, msg):
                    try:
                        d = json.loads(msg.data)
                        state = str(d.get("state", "—"))
                        self_.data.update(fd_status=state)
                        if state == "DONE":
                            self_.data.add_log("Feature detection complete", "info")
                        elif state == "ERROR":
                            self_.data.add_log("Feature detection error", "error")
                        elif state == "RUNNING":
                            self_.data.add_log("Feature detection started…", "info")
                    except Exception as e:
                        self_.data.add_log(f"Bad status message: {e}", "warn")

            node = _Inner(self.data)
            self.data.add_log("ROS2 node started — listening for topics", "info")
            while not self._stop.is_set():
                rclpy.spin_once(node, timeout_sec=0.05)
            node.destroy_node()
            rclpy.shutdown()

        except Exception as e:
            self.data.add_log(f"ROS2 init failed: {e}", "error")
            while not self._stop.is_set():
                time.sleep(0.5)

    def stop(self):
        self._stop.set()


# ─────────────────────── UTILITY ─────────────────────────────────────
def _quat_to_euler(x, y, z, w):
    roll  = math.degrees(math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    sinp  = 2*(w*y - z*x)
    pitch = math.degrees(math.copysign(math.pi/2, sinp)
                         if abs(sinp) >= 1 else math.asin(sinp))
    yaw   = math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return roll, pitch, yaw


# ─────────────────────── CUSTOM WIDGETS ──────────────────────────────

class SectionLabel(QLabel):
    def __init__(self, text):
        super().__init__(text.upper())
        self.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        self.setStyleSheet(f"color: {C_ORANGE.name()}; letter-spacing: 2px;")


class ValueLabel(QLabel):
    def __init__(self, text="—", size=14):
        super().__init__(text)
        self.setFont(QFont("Consolas", size, QFont.Weight.Bold))
        self.setStyleSheet(f"color: {C_TEXT.name()};")

    def set_colored(self, text, color: QColor):
        self.setText(text)
        self.setStyleSheet(f"color: {color.name()};")


class KeyLabel(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setFont(QFont("Consolas", 9))
        self.setStyleSheet(f"color: {C_MUTED.name()};")


class Card(QFrame):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            f"background: {C_CARD.name()}; border-radius: 8px;"
            f"border: 1px solid {C_BORDER.name()};"
        )
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 10, 12, 10)
        self._layout.setSpacing(8)
        if title:
            self._layout.addWidget(SectionLabel(title))

    def body(self):
        return self._layout


# ─── Artificial Horizon ──────────────────────────────────────────────
class AHI(QWidget):
    def __init__(self):
        super().__init__()
        self.roll = 0.0
        self.pitch = 0.0
        self.setMinimumSize(160, 160)

    def set_attitude(self, roll, pitch):
        self.roll = roll
        self.pitch = pitch
        self.update()

    def paintEvent(self, _):
        W, H = self.width(), self.height()
        r = min(W, H) / 2 - 4
        cx, cy = W / 2, H / 2
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        clip = QPainterPath()
        clip.addEllipse(QPointF(cx, cy), r, r)
        p.setClipPath(clip)
        p.fillRect(0, 0, W, H, QBrush(QColor("#1B3A6B")))

        p.save()
        p.translate(cx, cy)
        p.rotate(self.roll)
        pitch_px = self.pitch * r / 45.0
        p.fillRect(int(-r*2), int(pitch_px), int(r*4), int(r*4),
                   QBrush(QColor("#5C3317")))
        p.setPen(QPen(QColor("#CCCCCC"), 1.5))
        p.drawLine(int(-r), int(pitch_px), int(r), int(pitch_px))
        p.restore()

        p.setClipping(False)
        pen = QPen(C_ORANGE, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(int(cx - r*0.45), int(cy), int(cx - r*0.12), int(cy))
        p.drawLine(int(cx + r*0.12), int(cy), int(cx + r*0.45), int(cy))
        p.setPen(QPen(C_ORANGE, 2))
        p.drawEllipse(QPointF(cx, cy), 4, 4)

        p.setPen(QPen(C_BORDER, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)

        p.save()
        p.translate(cx, cy)
        p.rotate(self.roll)
        p.setPen(QPen(C_AMBER, 2))
        p.drawLine(0, int(-r), 0, int(-r + 10))
        p.restore()


# ─── Ground Map ──────────────────────────────────────────────────────
class MapPanel(QWidget):
    ARENA_W_M = 5
    ARENA_H_M = 10
    BUFFER_M  = 1
    TRAIL_MAX = None  # None = unlimited; keep the full flight path visible

    def __init__(self):
        super().__init__()
        self.setMinimumSize(280, 220)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.armed = False
        self.trail = []

    def set_pose(self, x, y, yaw, armed):
        self.x, self.y, self.yaw, self.armed = x, y, yaw, armed
        self.trail.append((x, y))
        if self.TRAIL_MAX is not None and len(self.trail) > self.TRAIL_MAX:
            self.trail.pop(0)
        self.update()

    def reset_trail(self):
        self.trail = []
        self.update()

    def _to_px(self, x_m, y_m, ox, oy, scale):
        return ox - x_m * scale*(-1), oy - y_m * scale

    def paintEvent(self, _):
        W, H = self.width(), self.height()
        margin = 18
        total_w = self.ARENA_W_M + 2 * self.BUFFER_M
        total_h = self.ARENA_H_M + 2 * self.BUFFER_M
        scale = min((W - 2*margin) / total_w, (H - 2*margin) / total_h)

        total_px_w = total_w * scale
        total_px_h = total_h * scale
        view_left = margin + ((W - 2*margin) - total_px_w) / 2
        view_top  = margin + ((H - 2*margin) - total_px_h) / 2
        buf_px = self.BUFFER_M * scale

        ox = view_left + total_px_w - buf_px
        oy = view_top  + total_px_h - buf_px
        arena_tlx = ox - self.ARENA_W_M * scale
        arena_tly = oy - self.ARENA_H_M * scale
        arena_w_px = self.ARENA_W_M * scale
        arena_h_px = self.ARENA_H_M * scale
        buf_tlx, buf_tly = view_left, view_top
        buf_brx = view_left + total_px_w
        buf_bry = view_top  + total_px_h

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(0, 0, W, H, QBrush(QColor("#15191C")))

        p.setBrush(QBrush(QColor(180, 30, 30, 18)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(QRectF(buf_tlx, buf_tly, total_px_w, total_px_h))

        hatch_pen = QPen(QColor(200, 50, 50, 35), 1)
        p.setPen(hatch_pen)
        step = 12
        x0, x1 = int(buf_tlx), int(buf_brx)
        y0, y1 = int(buf_tly), int(buf_bry)
        for i in range(0, (x1-x0)+(y1-y0), step):
            p.drawLine(QPointF(min(x0+i, x1), y0), QPointF(x0, min(y0+i, y1)))

        p.setBrush(QBrush(QColor("#15191C")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(QRectF(arena_tlx, arena_tly, arena_w_px, arena_h_px))

        p.setPen(QPen(QColor("#2A3338"), 1))
        gx = 0.0
        while gx <= self.ARENA_W_M + 1e-6:
            px, _ = self._to_px(gx, 0, ox, oy, scale)
            p.drawLine(QPointF(px, arena_tly), QPointF(px, oy))
            gx += 1.0
        gy = 0.0
        while gy <= self.ARENA_H_M + 1e-6:
            _, py = self._to_px(0, gy, ox, oy, scale)
            p.drawLine(QPointF(arena_tlx, py), QPointF(ox, py))
            gy += 1.0

        p.setPen(QPen(C_BORDER, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(arena_tlx, arena_tly, arena_w_px, arena_h_px))

        p.setPen(QPen(QColor(200, 60, 60, 180), 1, Qt.PenStyle.DashLine))
        p.drawRect(QRectF(buf_tlx, buf_tly, total_px_w, total_px_h))

        p.setFont(QFont("Consolas", 7))
        p.setPen(QPen(QColor(200, 60, 60, 140)))
        for tx, ty in [(buf_tlx+3, buf_tly+10), (buf_brx-72, buf_tly+10),
                       (buf_tlx+3, buf_bry-4),  (buf_brx-72, buf_bry-4)]:
            p.drawText(QPointF(tx, ty), "OUT OF BOUNDS")

        p.setPen(QPen(C_MUTED, 1))
        p.setFont(QFont("Consolas", 8))
        p.drawText(QPointF(arena_tlx+4, arena_tly+12), f"{self.ARENA_W_M:.1f} m")
        p.drawText(QPointF(arena_tlx+4, oy-4), "0")
        p.drawText(QPointF(ox-36, arena_tly+12), f"{self.ARENA_H_M:.1f} m")

        p.setPen(QPen(C_ORANGE, 2))
        p.setBrush(QBrush(C_ORANGE))
        p.drawEllipse(QPointF(ox, oy), 4, 4)
        p.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        p.setPen(QPen(C_ORANGE, 1))
        p.drawText(QPointF(ox-50, oy-6), "ORIGIN (0,0)")

        if len(self.trail) > 1:
            p.setPen(QPen(C_BLUE, 2))
            pts = [self._to_px(tx, ty, ox, oy, scale) for tx, ty in self.trail]
            for i in range(len(pts) - 1):
                p.drawLine(QPointF(*pts[i]), QPointF(*pts[i+1]))

        dpx, dpy = self._to_px(self.x, self.y, ox, oy, scale)
        in_arena = (0 <= self.x <= self.ARENA_W_M and
                    0 <= self.y <= self.ARENA_H_M)
        in_buf   = (not in_arena and
                    -self.BUFFER_M <= self.x <= self.ARENA_W_M + self.BUFFER_M and
                    -self.BUFFER_M <= self.y <= self.ARENA_H_M + self.BUFFER_M)
        if not in_arena and not in_buf:
            drone_col = C_RED
        elif not in_arena:
            drone_col = C_AMBER
        elif self.armed:
            drone_col = C_GREEN
        else:
            drone_col = C_MUTED

        p.save()
        p.translate(dpx, dpy)
        p.rotate(90 - self.yaw)
        path = QPainterPath()
        path.moveTo(0, -10)
        path.lineTo(6, 7)
        path.lineTo(-6, 7)
        path.closeSubpath()
        p.setPen(QPen(drone_col, 1.5))
        p.setBrush(QBrush(drone_col))
        p.drawPath(path)
        p.restore()

        oob = "  ⚠ OUT OF BOUNDS" if not in_arena else ""
        p.setPen(QPen(C_RED if not in_arena else C_TEXT, 1))
        p.setFont(QFont("Consolas", 8))
        p.drawText(QPointF(6, H-6), f"X={self.x:+.2f}m  Y={self.y:+.2f}m{oob}")


# ─── MAVROS Mode Banner ───────────────────────────────────────────────
class ModeDisplay(QWidget):
    def __init__(self):
        super().__init__()
        self.mode = "—"
        self.setFixedHeight(54)

    def set_mode(self, mode: str):
        self.mode = mode
        self.update()

    def paintEvent(self, _):
        W, H = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = mode_color(self.mode)
        bg  = QColor(col.red(), col.green(), col.blue(), 30)
        p.setBrush(QBrush(bg)); p.setPen(QPen(col, 2))
        p.drawRoundedRect(1, 1, W-2, H-2, 6, 6)
        p.setPen(QPen(col))
        p.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
        p.drawText(0, 0, W, H, Qt.AlignmentFlag.AlignCenter, self.mode)


# ─── Event Log ───────────────────────────────────────────────────────
class EventLog(QWidget):
    COLORS = {"info": C_TEXT, "warn": C_AMBER, "error": C_RED}

    def __init__(self):
        super().__init__()
        self._entries = []
        self.setMinimumHeight(80)

    def set_entries(self, entries):
        self._entries = entries[-10:]
        self.update()

    def paintEvent(self, _):
        W, H = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(0, 0, W, H, QBrush(QColor("#161616")))
        fh = 14
        for i, (ts, msg, lvl) in enumerate(reversed(self._entries)):
            y = i * fh + fh
            if y > H:
                break
            col = self.COLORS.get(lvl, C_TEXT)
            p.setPen(QPen(C_MUTED)); p.setFont(QFont("Consolas", 8))
            p.drawText(4, y-2, 56, fh, Qt.AlignmentFlag.AlignLeft, ts)
            p.setPen(QPen(col))
            p.drawText(60, y-2, W-64, fh, Qt.AlignmentFlag.AlignLeft, msg)


# ─── Feature Detection — single result row ────────────────────────────
class FeatureResultRow(QFrame):
    """
    One matched detection. Shows a small thumbnail (decoded from the
    base64 JPEG crop embedded in the result message, if present),
    the matched reference name, confidence, source image, and x/y/z.

    Designed to degrade gracefully: if no thumbnail data is present, or
    it fails to decode, a plain placeholder box is shown instead and
    everything else still renders normally.
    """
    THUMB_SIZE = 56

    def __init__(self, result: dict):
        super().__init__()
        self.setStyleSheet(
            f"background: {C_PANEL.name()}; border-radius: 6px;"
            f"border: 1px solid {C_BORDER.name()};"
        )
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 6, 6, 6)
        h.setSpacing(8)

        thumb_lbl = QLabel()
        thumb_lbl.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE)
        thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_lbl.setStyleSheet(
            f"background: {C_CARD.name()}; border-radius: 4px;"
            f"border: 1px solid {C_BORDER.name()};"
        )
        pix = self._load_thumb(result.get("thumb_b64"))
        if pix is not None:
            thumb_lbl.setPixmap(pix)
        else:
            thumb_lbl.setText("—")
            thumb_lbl.setStyleSheet(
                thumb_lbl.styleSheet() + f"color: {C_MUTED.name()}; font-size: 9pt;"
            )
        h.addWidget(thumb_lbl)

        info = QVBoxLayout()
        info.setSpacing(2)

        ref_name = str(result.get("ref", "—"))
        score = result.get("score")
        try:
            score_txt = f"{float(score) * 100:.1f}%"
            score_col = (C_GREEN if float(score) >= 0.85
                         else C_AMBER if float(score) >= 0.70 else C_TEXT)
        except (TypeError, ValueError):
            score_txt = "—"
            score_col = C_MUTED

        top_row = QHBoxLayout()
        name_lbl = QLabel(ref_name)
        name_lbl.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        name_lbl.setStyleSheet(f"color: {C_TEXT.name()};")
        score_lbl = QLabel(score_txt)
        score_lbl.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        score_lbl.setStyleSheet(f"color: {score_col.name()};")
        top_row.addWidget(name_lbl)
        top_row.addStretch()
        top_row.addWidget(score_lbl)
        info.addLayout(top_row)

        src_lbl = QLabel(str(result.get("image", "—")))
        src_lbl.setFont(QFont("Consolas", 8))
        src_lbl.setStyleSheet(f"color: {C_MUTED.name()};")
        info.addWidget(src_lbl)

        x, y, z = result.get("x"), result.get("y"), result.get("z")
        def fmt(v):
            try:
                return f"{float(v):+.2f}"
            except (TypeError, ValueError):
                return "—"
        coord_lbl = QLabel(f"x={fmt(x)}  y={fmt(y)}  z={fmt(z)}")
        coord_lbl.setFont(QFont("Consolas", 8))
        coord_lbl.setStyleSheet(f"color: {C_BLUE.name()};")
        info.addWidget(coord_lbl)

        h.addLayout(info, 1)

    def _load_thumb(self, thumb_b64):
        """Decode a base64 JPEG crop into a QPixmap. Returns None on any failure
        (missing field, bad base64, corrupt image data) — caller falls back
        to a placeholder so a bad/missing thumbnail never breaks the row."""
        if not thumb_b64:
            return None
        try:
            raw = base64.b64decode(thumb_b64)
            pix = QPixmap()
            ok = pix.loadFromData(raw)
            if not ok or pix.isNull():
                return None
            return pix.scaled(
                self.THUMB_SIZE, self.THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        except Exception:
            return None


# ─── Feature Detection — full panel ────────────────────────────────────
class FeatureDetectionPanel(QFrame):
    """
    Transfer status + detection status + progress + scrollable results list.
    Pure display widget — refresh(data) is called every tick from the
    GUI's existing 50ms timer, same as every other widget.
    """
    TRANSFER_LABELS = {
        0: ("INFLIGHT", C_MUTED),
        1: ("TRANSFERRING…", C_AMBER),
        2: ("TRANSFER COMPLETE", C_GREEN),
    }
    STATUS_COLORS = {
        "RUNNING": C_BLUE,
        "DONE":    C_GREEN,
        "ERROR":   C_RED,
        "—":       C_MUTED,
    }

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            f"background: {C_CARD.name()}; border-radius: 8px;"
            f"border: 1px solid {C_BORDER.name()};"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)
        outer.addWidget(SectionLabel("Feature Detection"))

        # Transfer status line
        self._transfer_lbl = QLabel("Transfer: —")
        self._transfer_lbl.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        self._transfer_lbl.setStyleSheet(f"color: {C_MUTED.name()};")
        outer.addWidget(self._transfer_lbl)

        # Detection status badge + current image
        status_row = QHBoxLayout()
        self._status_badge = QLabel("—")
        self._status_badge.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        self._status_badge.setStyleSheet(
            f"color: {C_MUTED.name()}; padding: 2px 8px;"
            f"border: 1px solid {C_MUTED.name()}; border-radius: 4px;"
        )
        status_row.addWidget(self._status_badge)
        self._current_img_lbl = QLabel("—")
        self._current_img_lbl.setFont(QFont("Consolas", 8))
        self._current_img_lbl.setStyleSheet(f"color: {C_MUTED.name()};")
        status_row.addWidget(self._current_img_lbl)
        status_row.addStretch()
        outer.addLayout(status_row)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFixedHeight(18)
        self._set_progress_style(C_BLUE)
        outer.addWidget(self._progress)

        # Scrollable results list
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFixedHeight(220)
        self._scroll.setStyleSheet("background: transparent; border: none;")
        self._results_container = QWidget()
        self._results_layout = QVBoxLayout(self._results_container)
        self._results_layout.setSpacing(6)
        self._results_layout.addStretch()
        self._scroll.setWidget(self._results_container)
        outer.addWidget(self._scroll)

        self._rendered_count = 0  # avoids rebuilding rows that already exist

    def _set_progress_style(self, color: QColor):
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: {C_BORDER.name()}; border-radius: 4px;
                text-align: center; color: {C_TEXT.name()};
                font-family: Consolas; font-size: 9px;
            }}
            QProgressBar::chunk {{ background: {color.name()}; border-radius: 4px; }}
        """)

    def refresh(self, data):
        # Transfer status
        label, col = self.TRANSFER_LABELS.get(
            data.transfer_status, ("—", C_MUTED))
        self._transfer_lbl.setText(f"Transfer: {label}")
        self._transfer_lbl.setStyleSheet(f"color: {col.name()};")

        # Detection status badge
        s_col = self.STATUS_COLORS.get(data.fd_status, C_MUTED)
        self._status_badge.setText(data.fd_status)
        self._status_badge.setStyleSheet(
            f"color: {s_col.name()}; padding: 2px 8px;"
            f"border: 1px solid {s_col.name()}; border-radius: 4px;"
        )
        self._current_img_lbl.setText(
            data.fd_current_image if data.fd_status == "RUNNING" else "")

        # Progress bar
        total = max(data.fd_progress_total, 1)
        self._progress.setRange(0, total)
        self._progress.setValue(min(data.fd_progress_current, total))
        self._set_progress_style(s_col if data.fd_status == "RUNNING" else C_BLUE)

        # Results list — only append new rows, never rebuild existing ones
        results = data.fd_results
        if len(results) > self._rendered_count:
            new_items = results[self._rendered_count:]
            # insert before the trailing stretch
            insert_at = self._results_layout.count() - 1
            for item in new_items:
                try:
                    row = FeatureResultRow(item)
                except Exception:
                    continue
                self._results_layout.insertWidget(insert_at, row)
                insert_at += 1
            self._rendered_count = len(results)


# ─────────────────────── MAIN WINDOW ─────────────────────────────────
class GroundStation(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASCEND Ground Station — Team Anveshak")
        self.setMinimumSize(1100, 740)
        self.data = DroneData()

        self._apply_global_style()
        self._build_ui()

        self.ros_worker = Ros2Worker(self.data)
        self.ros_worker.start()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(50)

    def _apply_global_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {C_BG.name()};
                color: {C_TEXT.name()};
                font-family: Consolas, monospace;
            }}
            QScrollBar:vertical {{
                background: {C_CARD.name()}; width: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {C_BORDER.name()}; border-radius: 3px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)

    # ── UI construction ───────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(8)

        outer.addWidget(self._build_header())
        outer.addWidget(self._build_mode_strip())

        body = QHBoxLayout()
        body.setSpacing(8)
        body.addLayout(self._build_left_col(),  3)
        body.addLayout(self._build_mid_col(),   4)
        body.addLayout(self._build_right_col(), 3)
        outer.addLayout(body, 1)

    def _build_header(self):
        w = QWidget()
        w.setStyleSheet(f"background: {C_PANEL.name()}; border-radius: 8px;")
        h = QHBoxLayout(w)
        h.setContentsMargins(12, 8, 12, 8)

        logo = QLabel("▲ ASCEND")
        logo.setFont(QFont("Consolas", 16, QFont.Weight.Bold))
        logo.setStyleSheet(f"color: {C_ORANGE.name()};")
        h.addWidget(logo)

        team = QLabel("Team Anveshak")
        team.setFont(QFont("Consolas", 9))
        team.setStyleSheet(f"color: {C_MUTED.name()};")
        h.addWidget(team)
        h.addStretch()

        self._pill_mavros  = self._make_pill("MAVROS",  False)
        self._pill_battery = self._make_pill("BATTERY", False)
        h.addWidget(self._pill_mavros)
        h.addWidget(self._pill_battery)

        self._arm_label = QLabel("DISARMED")
        self._arm_label.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self._arm_label.setStyleSheet(
            f"color: {C_MUTED.name()}; padding: 3px 10px;"
            f"border: 1px solid {C_MUTED.name()}; border-radius: 4px;"
        )
        h.addWidget(self._arm_label)

        self._clock = QLabel("--:--:--")
        self._clock.setFont(QFont("Consolas", 11))
        self._clock.setStyleSheet(f"color: {C_MUTED.name()}; padding-left:12px;")
        h.addWidget(self._clock)
        return w

    def _make_pill(self, text, active):
        lbl = QLabel(f"● {text}")
        lbl.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        col = C_GREEN if active else C_MUTED
        lbl.setStyleSheet(
            f"color: {col.name()}; padding: 3px 8px;"
            f"border: 1px solid {col.name()}; border-radius: 4px; margin-right: 4px;"
        )
        return lbl

    def _update_pill(self, lbl, text, active):
        col = C_GREEN if active else C_RED
        lbl.setText(f"● {text}")
        lbl.setStyleSheet(
            f"color: {col.name()}; padding: 3px 8px;"
            f"border: 1px solid {col.name()}; border-radius: 4px; margin-right: 4px;"
        )

    def _build_mode_strip(self):
        self._mode_display = ModeDisplay()
        self._mode_display.setStyleSheet(
            f"background: {C_PANEL.name()}; border-radius: 6px;")
        return self._mode_display

    def _btn_style(self, color: QColor):
        return f"""
            QPushButton {{
                background: transparent;
                color: {color.name()};
                border: 1px solid {color.name()};
                border-radius: 5px;
                padding: 6px 10px;
                font-family: Consolas; font-size: 9pt;
            }}
            QPushButton:hover   {{ background: rgba({color.red()},{color.green()},{color.blue()},30); }}
            QPushButton:pressed {{ background: rgba({color.red()},{color.green()},{color.blue()},60); }}
        """

    # ── Left: Attitude + Velocity + Detection ─────────────────────────
    def _build_left_col(self):
        col = QVBoxLayout()
        col.setSpacing(8)

        ahi_card = Card("Attitude")
        self._ahi = AHI()
        ahi_card.body().addWidget(self._ahi, alignment=Qt.AlignmentFlag.AlignCenter)

        g = QGridLayout(); g.setSpacing(6)
        self._roll_lbl  = ValueLabel("—")
        self._pitch_lbl = ValueLabel("—")
        self._yaw_lbl   = ValueLabel("—")
        self._alt_lbl   = ValueLabel("—")
        for row, (key, val) in enumerate([
            ("ROLL",  self._roll_lbl),
            ("PITCH", self._pitch_lbl),
            ("YAW",   self._yaw_lbl),
            ("ALT",   self._alt_lbl),
        ]):
            g.addWidget(KeyLabel(key), row, 0)
            g.addWidget(val, row, 1)
        ahi_card.body().addLayout(g)
        col.addWidget(ahi_card)

        vel_card = Card("Velocity")
        vg = QGridLayout(); vg.setSpacing(6)
        self._vx_lbl  = ValueLabel("—", 13)
        self._vy_lbl  = ValueLabel("—", 13)
        self._vz_lbl  = ValueLabel("—", 13)
        self._spd_lbl = ValueLabel("—", 13)
        for row, (key, val) in enumerate([
            ("Vx m/s",  self._vx_lbl),
            ("Vy m/s",  self._vy_lbl),
            ("Vz m/s",  self._vz_lbl),
            ("SPD m/s", self._spd_lbl),
        ]):
            vg.addWidget(KeyLabel(key), row, 0)
            vg.addWidget(val, row, 1)
        vel_card.body().addLayout(vg)
        col.addWidget(vel_card)

        col.addWidget(self._build_detection_card())
        col.addStretch()
        return col

    def _build_detection_card(self):
        card = Card("Field Detection")
        g = QGridLayout()
        g.setSpacing(8)
        g.addWidget(KeyLabel("ArUco"), 0, 0)
        self._aruco_lbl = ValueLabel("—", 12)
        g.addWidget(self._aruco_lbl, 0, 1)
        card.body().addLayout(g)

        self._aruco_img_lbl = QLabel("No debug image yet")
        self._aruco_img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._aruco_img_lbl.setFixedHeight(160)
        self._aruco_img_lbl.setStyleSheet(
            f"background: {C_PANEL.name()}; border-radius: 6px;"
            f"border: 1px solid {C_BORDER.name()}; color: {C_MUTED.name()};"
            f"font-family: Consolas; font-size: 9pt;"
        )
        self._aruco_img_lbl.setScaledContents(False)
        card.body().addWidget(self._aruco_img_lbl)
        self._aruco_img_jpeg_seen = None  # last bytes id rendered, avoids re-decoding unchanged frames
        return card

    def _update_aruco_image(self, jpeg_bytes):
        """Render the latest /aruco_debug_image/compress frame.
        Falls back to the placeholder text on missing/corrupt data so a bad
        frame never breaks the rest of the panel."""
        if jpeg_bytes is None:
            return
        if jpeg_bytes is self._aruco_img_jpeg_seen:
            return  # same object as last tick — nothing changed, skip decode
        self._aruco_img_jpeg_seen = jpeg_bytes
        try:
            pix = QPixmap()
            ok = pix.loadFromData(jpeg_bytes)
            if not ok or pix.isNull():
                raise ValueError("decode failed")
            scaled = pix.scaled(
                self._aruco_img_lbl.width() or 260,
                self._aruco_img_lbl.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._aruco_img_lbl.setPixmap(scaled)
        except Exception:
            self._aruco_img_lbl.setText("Bad frame — waiting for next…")

    # ── Middle: Mode + Position + Map + Fallbacks ─────────────────────
    def _build_mid_col(self):
        col = QVBoxLayout()
        col.setSpacing(8)

        mode_card = Card("Flight Mode (MAVROS)")
        self._mode_name = QLabel("—")
        self._mode_name.setFont(QFont("Consolas", 22, QFont.Weight.Bold))
        self._mode_name.setStyleSheet(f"color: {C_MUTED.name()};")
        self._mode_sub = QLabel("Waiting for connection…")
        self._mode_sub.setFont(QFont("Consolas", 9))
        self._mode_sub.setStyleSheet(f"color: {C_MUTED.name()};")
        self._mode_sub.setWordWrap(True)
        mode_card.body().addWidget(self._mode_name)
        mode_card.body().addWidget(self._mode_sub)
        col.addWidget(mode_card)

        pos_card = Card("Position — relative to origin (ENU)")
        pg = QGridLayout(); pg.setSpacing(6)
        self._px = ValueLabel("—", 13)
        self._py = ValueLabel("—", 13)
        self._pz = ValueLabel("—", 13)
        for i, (k, v) in enumerate([
            ("X (East)",  self._px),
            ("Y (North)", self._py),
            ("Z (Up)",    self._pz),
        ]):
            pg.addWidget(KeyLabel(k), i, 0)
            pg.addWidget(v, i, 1)
            pg.addWidget(KeyLabel("m"), i, 2)
        pos_card.body().addLayout(pg)
        col.addWidget(pos_card)

        map_card = Card("Flight Arena Map")
        self._map = MapPanel()
        map_card.body().addWidget(self._map)
        col.addWidget(map_card)

        em_card = Card("Emergency Fallbacks")
        em_card.setStyleSheet(
            f"background: #1F1010; border: 1px solid {C_RED.darker(120).name()};"
            f"border-radius: 8px;"
        )
        fg = QGridLayout(); fg.setSpacing(4)
        for i, (trigger, action) in enumerate(FALLBACKS.items()):
            k = QLabel(trigger); k.setFont(QFont("Consolas", 8))
            k.setStyleSheet(f"color: {C_AMBER.name()};")
            v = QLabel(action); v.setFont(QFont("Consolas", 8))
            v.setStyleSheet(f"color: {C_RED.name()};")
            fg.addWidget(k, i, 0); fg.addWidget(v, i, 1)
        em_card.body().addLayout(fg)
        col.addWidget(em_card)

        col.addStretch()
        return col

    # ── Right: Total Voltage + Origin + Log ───────────────────────────
    def _build_right_col(self):
        col = QVBoxLayout()
        col.setSpacing(8)

        # Total voltage only
        batt_card = Card("Battery")
        tg = QGridLayout(); tg.setSpacing(6)
        self._total_v_lbl = ValueLabel("—", 14)
        tg.addWidget(KeyLabel("Total V"), 0, 0)
        tg.addWidget(self._total_v_lbl, 0, 1)
        batt_card.body().addLayout(tg)
        col.addWidget(batt_card)

        # Origin control card
        origin_card = Card("Odometry Origin")
        self._origin_lbl = QLabel("Raw: (—, —, —)")
        self._origin_lbl.setFont(QFont("Consolas", 8))
        self._origin_lbl.setStyleSheet(f"color: {C_MUTED.name()};")
        self._origin_lbl.setWordWrap(True)
        origin_card.body().addWidget(self._origin_lbl)

        set_origin_btn = QPushButton("⊕  Set Origin Here")
        set_origin_btn.setStyleSheet(self._btn_style(C_ORANGE))
        set_origin_btn.clicked.connect(self._set_origin)
        origin_card.body().addWidget(set_origin_btn)

        reset_origin_btn = QPushButton("↺  Reset Origin to (0, 0, 0)")
        reset_origin_btn.setStyleSheet(self._btn_style(C_MUTED))
        reset_origin_btn.clicked.connect(self._reset_origin)
        origin_card.body().addWidget(reset_origin_btn)

        col.addWidget(origin_card)

        # Feature detection panel
        self._feature_panel = FeatureDetectionPanel()
        col.addWidget(self._feature_panel)

        # Event log
        log_card = Card("Event Log")
        self._log = EventLog()
        log_card.body().addWidget(self._log)
        col.addWidget(log_card, 1)

        col.addStretch()
        return col

    # ── Origin actions ────────────────────────────────────────────────
    def _set_origin(self):
        self.data.set_origin_here()
        self._map.reset_trail()

    def _reset_origin(self):
        with self.data._lock:
            self.data.origin_x = 0.0
            self.data.origin_y = 0.0
            self.data.origin_z = 0.0
        self._map.reset_trail()
        self.data.add_log("Origin reset to (0, 0, 0)", "info")

    # ── Refresh (50 ms) ───────────────────────────────────────────────
    def _refresh(self):
        d = self.data.snapshot()

        self._update_pill(self._pill_mavros,  "MAVROS",  d.connected_mavros)
        self._update_pill(self._pill_battery, "BATTERY", d.connected_battery)

        arm_col = C_GREEN if d.armed else C_MUTED
        self._arm_label.setText("ARMED" if d.armed else "DISARMED")
        self._arm_label.setStyleSheet(
            f"color: {arm_col.name()}; padding: 3px 10px;"
            f"border: 1px solid {arm_col.name()}; border-radius: 4px;"
        )

        # Mode
        col = mode_color(d.mode)
        self._mode_name.setText(d.mode)
        self._mode_name.setStyleSheet(f"color: {col.name()};")
        self._mode_display.set_mode(d.mode)
        _hints = {
            "MANUAL":       "Manual RC control — no position hold",
            "STABILIZED":   "Stabilised attitude — no position hold",
            "ALTCTL":       "Altitude hold active",
            "POSCTL":       "Position hold active",
            "OFFBOARD":     "Offboard control — receiving setpoints",
            "AUTO.TAKEOFF": f"Auto takeoff · alt = {d.z:.2f} m",
            "AUTO.LOITER":  "Loitering at current position",
            "AUTO.RTL":     "Return to launch in progress",
            "AUTO.LAND":    "AUTO.LAND engaged · descending",
            "AUTO.MISSION": "Executing mission plan",
        }
        self._mode_sub.setText(_hints.get(d.mode, ""))

        # Attitude
        self._ahi.set_attitude(d.roll, d.pitch)
        def fmt_angle(v, warn, crit):
            c = C_RED if abs(v) >= crit else C_AMBER if abs(v) >= warn else C_TEXT
            return f"{v:+.2f}°", c
        r_t, r_c = fmt_angle(d.roll,  20, 35)
        p_t, p_c = fmt_angle(d.pitch, 15, 30)
        self._roll_lbl.set_colored(r_t, r_c)
        self._pitch_lbl.set_colored(p_t, p_c)
        self._yaw_lbl.set_colored(f"{d.yaw:.1f}°", C_TEXT)
        self._alt_lbl.set_colored(f"{d.z:.2f} m",
                                   C_GREEN if d.z > 0.3 else C_TEXT)

        # Velocity
        def vel_col(v):
            return C_RED if abs(v) > 3.0 else C_AMBER if abs(v) > 1.5 else C_BLUE
        self._vx_lbl.set_colored(f"{d.vx:+.2f}", vel_col(d.vx))
        self._vy_lbl.set_colored(f"{d.vy:+.2f}", vel_col(d.vy))
        self._vz_lbl.set_colored(f"{d.vz:+.2f}", vel_col(d.vz))
        self._spd_lbl.set_colored(f"{d.speed:.2f}", C_BLUE)

        # Position (relative)
        self._px.set_colored(f"{d.x:+.3f}", C_TEXT)
        self._py.set_colored(f"{d.y:+.3f}", C_TEXT)
        self._pz.set_colored(f"{d.z:+.3f}", C_TEXT)

        # Map
        self._map.set_pose(d.x, d.y, d.yaw, d.armed)

        # Origin readout
        self._origin_lbl.setText(
            f"Raw: ({d.raw_x:.2f}, {d.raw_y:.2f}, {d.raw_z:.2f})\n"
            f"Origin: ({d.origin_x:.2f}, {d.origin_y:.2f}, {d.origin_z:.2f})"
        )

        # Battery — total voltage only
        tv_col = C_RED if d.total_v < 13.5 else C_AMBER if d.total_v < 14.4 else C_GREEN
        self._total_v_lbl.set_colored(f"{d.total_v:.2f}V", tv_col)

        # Field detection
        self._aruco_lbl.set_colored(
            "DETECTED" if d.aruco_detected else "NOT FOUND",
            C_GREEN if d.aruco_detected else C_MUTED,
        )
        self._update_aruco_image(d.aruco_debug_jpeg)

        self._log.set_entries(d.log_entries)
        self._feature_panel.refresh(d)

    def closeEvent(self, e):
        self.ros_worker.stop()
        self.ros_worker.wait(2000)
        super().closeEvent(e)


# ─────────────────────────────── MAIN ────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ASCEND Ground Station")
    win = GroundStation()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()