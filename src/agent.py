# ==============================================================================
# Self-Driving Agent for Racing Games
#
# Author: [Your Name/Alias]
# Version: 5.2.0
#
# Description:
# This script implements a self-driving agent that uses computer vision to
# navigate a race track in a video game. It captures the screen, processes
# the image to detect lane lines, and computes steering commands using a PID
# controller. The agent features a web-based dashboard for real-time
# monitoring and control, along with an automated PID tuning system using a
# genetic algorithm.
#
# This version preserves the original, complex perception logic while making
# key parameters resolution-independent and adding new UI-driven features.
# ==============================================================================

import cv2
import numpy as np
from mss import mss
import pyautogui
import keyboard
import time
from flask import Flask, Response, request, render_template, jsonify
from threading import Thread, Lock
from waitress import serve
import sys
import os
from datetime import datetime
from collections import deque
import csv
import socket
import json
from scipy.spatial import distance as dist
from collections import OrderedDict
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import logging
import pandas as pd

# ==============================================================================
# --- Logging Configuration ---
# ==============================================================================

# Suppress scikit-learn warnings for RANSAC on small sample sizes
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 1. Create a custom logger separate from the root logger
log = logging.getLogger('AgentLogger')
log.setLevel(logging.INFO)
log.propagate = False  # Prevent messages from propagating to the root logger

# 2. Create a file handler to write detailed perception logs
file_handler = logging.FileHandler("agent.log", mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
log.addHandler(file_handler)

# 3. Create a console handler for user-facing messages, can be toggled
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(message)s'))
# Console logging is off by default, enabled via UI for diagnostics

# ==============================================================================
# --- Genetic Algorithm PID Tuner Configuration ---
# ==============================================================================
GENETIC_ALGORITHM_CONFIG = {
    "population_size": 10,
    "num_generations": 5,
    "duration_per_test": 15,
    "mutation_rate": 0.2,
    "mutation_strength": 0.3,
    "search_space": {
        "kp": (0.01, 1.0),
        "ki": (0.0001, 0.1),
        "kd": (0.01, 2.0)
    }
}

# ==============================================================================
# --- Core Agent & Performance Configuration ---
# ==============================================================================
INITIAL_KP = 0.08
INITIAL_KI = 0.001
INITIAL_KD = 0.8
INITIAL_STEERING_GAIN = 4.0
PWM_FREQUENCY = 75
FRAME_RATE = 30
JPEG_QUALITY = 50

# --- Default ROI Values (as ratios of screen/game window size) ---
DEFAULT_GAME_BOUNDING_BOX_RATIOS = {'top': 0.34, 'left': 0.2734, 'width': 0.6, 'height': 0.5442}
DEFAULT_CAR_ROI_RECT_RATIOS = (0.1328, 0.2703, 0.49, 0.7175)
DEFAULT_LANE_ROI_RECT_RATIOS = (0.1111, 0.0949, 0.77, 0.1539)

# ==============================================================================
# --- Global Shared Variables & Control Flags ---
# ==============================================================================
last_processed_frame = None
frame_lock = Lock()
agent_running = False
shutdown_signal = False
control_signal = 0.0
control_lock = Lock()

# --- UI-Controllable Flags ---
DIAGNOSTICS_ON = False
VIDEO_RECORDING_ENABLED = False
LOGGING_ENABLED = True

# --- UI-Specific State Dictionary ---
ui_status = {
    "agent_status": "Paused",
    "best_score": 0.0,
    "best_pid": {"kp": INITIAL_KP, "ki": INITIAL_KI, "kd": INITIAL_KD},
    "tune_iterations": 0,
    "tuning_mode": "auto",
    "video_recording": VIDEO_RECORDING_ENABLED,
    "data_logging": LOGGING_ENABLED
}
ui_lock = Lock()

CONFIG_FILE = 'config.json'
config = {}


def load_config():
    """
    Loads agent configuration from a JSON file. If the file does not exist
    or is invalid, it returns a default configuration dictionary.
    """
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                print("INFO: Loading configuration from config.json")
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"WARNING: Could not load or parse config.json: {e}. Using defaults.")

    return {
        "pid": {"kp": INITIAL_KP, "ki": INITIAL_KI, "kd": INITIAL_KD},
        "best_pid": {"kp": INITIAL_KP, "ki": INITIAL_KI, "kd": INITIAL_KD},
        "best_ever_score": 0.0,
        "gain": INITIAL_STEERING_GAIN,
        "roi_ratios": {
            "game": DEFAULT_GAME_BOUNDING_BOX_RATIOS,
            "car": DEFAULT_CAR_ROI_RECT_RATIOS,
            "lane": DEFAULT_LANE_ROI_RECT_RATIOS
        }
    }


def save_config():
    """
    Saves the current agent state (PID, gain, ROIs) to the JSON config file.
    This function uses non-blocking lock acquisition to avoid deadlocks.
    """
    got_agent_lock = agent_instance.lock.acquire(blocking=False)
    if not got_agent_lock:
        return

    got_ui_lock = ui_lock.acquire(blocking=False)
    if not got_ui_lock:
        agent_instance.lock.release()
        return

    try:
        config_data = {
            "pid": {"kp": agent_instance.Kp, "ki": agent_instance.Ki, "kd": agent_instance.Kd},
            "best_pid": ui_status["best_pid"],
            "best_ever_score": ui_status.get("best_ever_score", 0.0),
            "gain": agent_instance.steering_gain,
            "roi_ratios": config.get('roi_ratios')
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)
    finally:
        agent_instance.lock.release()
        ui_lock.release()


def convert_ratios_to_pixels(game_ratios, car_ratios, lane_ratios):
    """
    Converts relative ROI ratios to absolute pixel values based on the primary
    monitor's screen resolution.
    """
    screen_width, screen_height = pyautogui.size()
    print(f"INFO: Detected screen resolution: {screen_width}x{screen_height}")

    game_box_px = {
        'left': int(screen_width * game_ratios['left']),
        'top': int(screen_height * game_ratios['top']),
        'width': int(screen_width * game_ratios['width']),
        'height': int(screen_height * game_ratios['height'])
    }

    car_roi_px = (
        int(game_box_px['width'] * car_ratios[0]),
        int(game_box_px['height'] * car_ratios[1]),
        int(game_box_px['width'] * car_ratios[2]),
        int(game_box_px['height'] * car_ratios[3])
    )
    lane_roi_px = (
        int(game_box_px['width'] * lane_ratios[0]),
        int(game_box_px['height'] * lane_ratios[1]),
        int(game_box_px['width'] * lane_ratios[2]),
        int(game_box_px['height'] * lane_ratios[3])
    )

    print(f"INFO: Calculated Game Box (pixels): {game_box_px}")
    return game_box_px, car_roi_px, lane_roi_px


class SelfDrivingAgent:
    """
    The core class for the self-driving agent. It encapsulates all logic for
    perception, decision-making (PID control), and state management.
    The perception logic within this class is preserved from the original script
    to ensure identical behavior.
    """

    def __init__(self, bounding_box, car_roi_rect, lane_roi_rect):
        self.lock = Lock()
        self.bounding_box = bounding_box
        self.car_roi_rect = car_roi_rect
        self.lane_roi_rect = lane_roi_rect

        # --- Steering Target ---
        screen_width, _ = pyautogui.size()
        screen_center_x = screen_width // 2
        self.region_center_x = screen_center_x - self.bounding_box['left']
        print(f"INFO: Screen Center Target (absolute): {screen_center_x}px")
        print(f"INFO: Game Window ROI starts at: {self.bounding_box['left']}px")
        print(f"INFO: New Relative Target (region_center_x): {self.region_center_x}px")

        # --- PID Controller State ---
        self.Kp, self.Ki, self.Kd = INITIAL_KP, INITIAL_KI, INITIAL_KD
        self.steering_gain = INITIAL_STEERING_GAIN
        self.integral, self.previous_error = 0, 0
        self.previous_time = time.time()
        self.derivative_filter_time_constant = 0.05
        self.filtered_derivative = 0.0
        self.tracking_gain = 0.1

        # --- Perception State & Scoring ---
        self.lane_center_x = None
        self.error_window = deque(maxlen=5)
        self.last_known_lane_center_x = None
        self.stability_score = 0
        self.score_window = deque(maxlen=100)
        self.lane_tracker = CentroidTracker(maxDisappeared=5)
        self.last_good_line_params = None
        self.lane_line = SmoothedLine(smoothing_factor=0.3)
        self.current_ransac_angle = 0.0
        self.last_validated_angle = None

        # --- Original Hardcoded Perception Parameters ---
        # These values are kept here to show the original state before scaling.
        # They are now dynamically calculated below.
        # self.match_threshold_pixels = 100
        # self.max_position_deviation_pixels = 80
        # self.dbscan_eps = 40
        # self.ransac_residual_threshold = 50
        # self.min_contour_area = 300
        # self.max_contour_area = 2000

        # --- Resolution-Independent Scaling ---
        # Calculate absolute pixel values from ratios based on the current ROI size.
        # This ensures the agent's perception behaves consistently on any monitor.
        # Reference resolution for ratios: 3840x2400
        game_w, game_h = self.bounding_box['width'], self.bounding_box['height']
        lane_w, lane_h = self.lane_roi_rect[2], self.lane_roi_rect[3]

        # Ratios derived from original values and a 3840x2400 screen
        # where lane ROI was approx 1774x201 pixels.
        self.match_threshold_pixels = int(lane_w * (100 / 1774.0))
        self.max_position_deviation_pixels = int(lane_w * (80 / 1774.0))
        self.dbscan_eps = int(lane_w * (40 / 1774.0))
        self.ransac_residual_threshold = int(lane_h * (50 / 201.0))
        lane_area = lane_w * lane_h
        self.min_contour_area = int(lane_area * (300 / (1774.0 * 201.0)))
        self.max_contour_area = int(lane_area * (2000 / (1774.0 * 201.0)))

        # --- Other Original Perception Parameters ---
        self.sanity_check_threshold = self.bounding_box['width'] * 0.1
        self.angle_change_threshold = 0.5
        self.max_angle_deviation = 0.96
        self.persistence_error_threshold = 3.0
        self.cluster_num_bins = 15
        self.cluster_window_size = 2
        self.dbscan_min_samples = 5
        self.max_angle_jump_degrees = 9.0
        self.blob_verticality_threshold = 0.8

    def _calculate_line_fit_error(self, line_params, points):
        """Calculates the average perpendicular distance of points from a line."""
        if line_params is None or len(points) == 0:
            return float('inf')
        vx, vy, x0, y0 = line_params
        distances = [abs(vx * (p[1] - y0) - vy * (p[0] - x0)) for p in points]
        return np.mean(distances)

    def _get_shape_intersections(self, contour):
        """
        Original complex function to generate a dense cloud of points from a contour.
        This logic is preserved exactly to maintain original behavior.
        """
        initial_points = []
        if len(contour) < 5:
            return []
        try:
            x_b, y_b, w_b, h_b = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            if M["m00"] == 0: return []
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            initial_points.append((cx, cy))
            line_params = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x, y = line_params[0][0], line_params[1][0], line_params[2][0], line_params[3][0]
            ellipse_params = cv2.fitEllipse(contour)
            (e_cx, e_cy), (ma, mi), angle_deg = ellipse_params
            a_semi_major, b_semi_minor = ma / 2.0, mi / 2.0
            circle_params = cv2.minEnclosingCircle(contour)
            (c_cx, c_cy), radius = circle_params
            p0_minus_c = np.array([x - c_cx, y - c_cy])
            v = np.array([vx, vy])
            quad_a = np.dot(v, v)
            quad_b = 2 * np.dot(p0_minus_c, v)
            quad_c = np.dot(p0_minus_c, p0_minus_c) - radius ** 2
            if abs(quad_a) > 1e-5:
                discriminant = quad_b ** 2 - 4 * quad_a * quad_c
                if discriminant >= 0:
                    sqrt_disc = np.sqrt(discriminant)
                    t1 = (-quad_b + sqrt_disc) / (2 * quad_a)
                    t2 = (-quad_b - sqrt_disc) / (2 * quad_a)
                    initial_points.append((int(x + t1 * vx), int(y + t1 * vy)))
                    initial_points.append((int(x + t2 * vx), int(y + t2 * vy)))
            if a_semi_major > 1e-5 and b_semi_minor > 1e-5:
                p0_translated = np.array([x - e_cx, y - e_cy])
                angle_rad = np.deg2rad(-angle_deg)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                p0_rot = rot_matrix @ p0_translated
                v_rot = rot_matrix @ v
                A = (v_rot[0] ** 2 / a_semi_major ** 2) + (v_rot[1] ** 2 / b_semi_minor ** 2)
                B = 2 * ((p0_rot[0] * v_rot[0] / a_semi_major ** 2) + (p0_rot[1] * v_rot[1] / b_semi_minor ** 2))
                C = (p0_rot[0] ** 2 / a_semi_major ** 2) + (p0_rot[1] ** 2 / b_semi_minor ** 2) - 1
                discriminant_e = B ** 2 - 4 * A * C
                if discriminant_e >= 0 and abs(A) > 1e-5:
                    sqrt_disc_e = np.sqrt(discriminant_e)
                    t1_e = (-B + sqrt_disc_e) / (2 * A)
                    t2_e = (-B - sqrt_disc_e) / (2 * A)
                    initial_points.append((int(x + t1_e * vx), int(y + t1_e * vy)))
                    initial_points.append((int(x + t2_e * vx), int(y + t2_e * vy)))
            if len(initial_points) < 2:
                return initial_points
            keypoints_arr = np.array(initial_points, dtype=np.float32)
            mini_line = cv2.fitLine(keypoints_arr, cv2.DIST_L2, 0, 0.01, 0.01)
            vx_mini, vy_mini = mini_line[0][0], mini_line[1][0]
            x_mini, y_mini = mini_line[2][0], mini_line[3][0]
            if abs(vx_mini) > abs(vy_mini) * self.blob_verticality_threshold:
                log.info(
                    f"Blob failed verticality check (vx: {vx_mini:.2f}, vy: {vy_mini:.2f}). Returning keypoints only.")
                return list(map(tuple, initial_points))
            final_points = list(map(tuple, initial_points))
            if h_b > w_b:
                start_y, end_y = y_b, y_b + h_b
                for interp_y in np.linspace(start_y, end_y, 10):
                    if abs(vy_mini) > 1e-5:
                        interp_x = ((interp_y - y_mini) * vx_mini / vy_mini) + x_mini
                        if x_b <= interp_x <= x_b + w_b:
                            final_points.append((int(interp_x), int(interp_y)))
            else:
                start_x, end_x = x_b, x_b + w_b
                for interp_x in np.linspace(start_x, end_x, 10):
                    if abs(vx_mini) > 1e-5:
                        interp_y = ((interp_x - x_mini) * vy_mini / vx_mini) + y_mini
                        if y_b <= interp_y <= y_b + h_b:
                            final_points.append((int(interp_x), int(interp_y)))
            return final_points
        except (cv2.error, ValueError):
            return initial_points

    def _dbscan_filter_points(self, points):
        """
        Performs DBSCAN clustering using the dynamically scaled `self.dbscan_eps`.
        """
        if len(points) < self.dbscan_min_samples:
            return points
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(points)
        labels = db.labels_
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(counts) == 0:
            log.warning("DBSCAN classified all points as noise.")
            return np.array([])
        largest_cluster_label = unique_labels[counts.argmax()]
        cluster_mask = (labels == largest_cluster_label)
        filtered_points = np.array(points)[cluster_mask]
        num_removed = len(points) - len(filtered_points)
        if num_removed > 0:
            log.info(f"DBSCAN Filter: Kept {len(filtered_points)}, removed {num_removed} outliers.")
        return filtered_points

    def _find_lane_contours(self, mask):
        """
        Finds contours and filters them by area using the dynamically scaled
        `self.min_contour_area` and `self.max_contour_area`.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                filtered_contours.append(contour)
        return filtered_contours

    def update_pid(self, Kp, Ki, Kd):
        """Safely updates the PID controller gains."""
        with self.lock:
            self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral, self.previous_error = 0, 0
        self.score_window.clear()
        print(f"Updated PID: Kp={self.Kp:.4f}, Ki={self.Ki:.4f}, Kd={self.Kd:.4f}")

    def update_gain(self, gain):
        """Safely updates the steering gain."""
        with self.lock:
            self.steering_gain = gain
        print(f"Updated Steering Gain: {self.steering_gain:.2f}")

    def _detect_lanes_dual_pipeline(self, original_screenshot):
        """
        The main perception pipeline, with logic identical to the original script.
        It now uses the dynamically scaled parameters for filtering and regression.
        """
        lx, ly, lw, lh = self.lane_roi_rect
        lane_roi_img = original_screenshot[ly:ly + lh, lx:lx + lw]

        debug_data = {
            'initial_points': 0, 'dbscan_points': 0, 'inliers': 0,
            'angle': 0.0, 'line_params': 'None', 'target_x': 0,
            'force_accept': False, 'is_stable': False, 'angle_jump_ok': True, 'validated': False
        }

        hsv = cv2.cvtColor(lane_roi_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 25, 255]))
        kernel = np.ones((3, 3), np.uint8)
        mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_erode = cv2.erode(mask_opened, kernel, iterations=3)
        mask_cleaned = cv2.dilate(mask_erode, kernel, iterations=5)

        contours = self._find_lane_contours(mask_cleaned)
        rects_for_tracker = [cv2.boundingRect(c) for c in contours]
        tracked_blobs = self.lane_tracker.update(rects_for_tracker)

        ransac_points = []
        for contour in contours:
            ransac_points.extend(self._get_shape_intersections(contour))
        if tracked_blobs:
            weight = 3
            for _ in range(weight):
                ransac_points.extend(list(tracked_blobs.values()))
        debug_data['initial_points'] = len(ransac_points)

        if len(ransac_points) > 0:
            ransac_points = self._dbscan_filter_points(ransac_points)
        debug_data['dbscan_points'] = len(ransac_points)

        # FIX for DeprecationWarning: Use explicit length check
        points = np.array([tuple(p) for p in ransac_points]) if len(ransac_points) > 0 else np.array([])

        self.lane_center_x = None
        validated_line_params = None
        inlier_mask = None
        ransac = RANSACRegressor(residual_threshold=self.ransac_residual_threshold, max_trials=100)

        if len(points) >= 2:
            try:
                X = points[:, 0].reshape(-1, 1)
                y_true = points[:, 1]
                ransac.fit(X, y_true)
                inlier_mask = ransac.inlier_mask_
                filtered_points = points[inlier_mask]
                debug_data['inliers'] = len(filtered_points)

                if len(filtered_points) >= 2:
                    slope = ransac.estimator_.coef_[0]
                    vec = np.array([1, slope])
                    vec_norm = np.linalg.norm(vec)
                    vx, vy = vec / vec_norm
                    cx, cy = np.mean(filtered_points, axis=0)
                    candidate_params = np.array([vx, vy, cx, cy])
                    current_angle = np.rad2deg(np.arctan2(vy, vx))
                    debug_data['angle'] = current_angle
                    debug_data['line_params'] = f"[{vx:.2f},{vy:.2f},{cx:.1f},{cy:.1f}]"

                    force_accept_new_line = False
                    if self.lane_line.line_params is not None:
                        new_line_error = self._calculate_line_fit_error(candidate_params, filtered_points)
                        old_line_error = self._calculate_line_fit_error(self.lane_line.line_params, filtered_points)
                        if old_line_error > new_line_error * self.persistence_error_threshold:
                            force_accept_new_line = True
                    debug_data['force_accept'] = force_accept_new_line

                    is_stable_for_jitter = True
                    if self.lane_line.line_params is not None:
                        prev_params = self.lane_line.line_params
                        angle_dot_product = abs(np.dot(candidate_params[:2], prev_params[:2]))
                        if angle_dot_product < self.max_angle_deviation: is_stable_for_jitter = False
                        if is_stable_for_jitter and abs(prev_params[1]) > 1e-5 and abs(candidate_params[1]) > 1e-5:
                            steering_horizon_y = lh * 0.8
                            prev_x = ((steering_horizon_y - prev_params[3]) * prev_params[0] / prev_params[1]) + \
                                     prev_params[2]
                            cand_x = ((steering_horizon_y - cy) * vx / vy) + cx
                            if abs(prev_x - cand_x) > self.max_position_deviation_pixels: is_stable_for_jitter = False
                    debug_data['is_stable'] = is_stable_for_jitter

                    angle_jump_ok = True
                    if self.last_validated_angle is not None:
                        angle_diff = abs(current_angle - self.last_validated_angle)
                        if angle_diff > self.max_angle_jump_degrees:
                            angle_jump_ok = False
                    debug_data['angle_jump_ok'] = angle_jump_ok

                    if force_accept_new_line or (is_stable_for_jitter and angle_jump_ok):
                        if abs(vy) > 1e-5:
                            steering_horizon_y = lh * 0.8
                            target_x = int(((steering_horizon_y - cy) * vx / vy) + cx)
                            debug_data['target_x'] = target_x
                            if 0 < target_x < lw:
                                validated_line_params = candidate_params
                                debug_data['validated'] = True
                                self.last_validated_angle = current_angle
            except Exception as e:
                if LOGGING_ENABLED: log.error(f"DBSCAN/RANSAC failed. Error: {e}", exc_info=False)
                pass

        if not debug_data['validated']:
            self.last_validated_angle = None

        if LOGGING_ENABLED:
            log_str = ", ".join([f"{k}={v}" for k, v in debug_data.items()])
            log.info(f"PERCEPTION_FRAME: {log_str}")
        self.current_ransac_angle = debug_data['angle']

        smoothed_line_params = self.lane_line.update(validated_line_params, force_update=debug_data['force_accept'])

        if smoothed_line_params is not None:
            vx, vy, x, y = smoothed_line_params
            if abs(vx) > 1e-5:
                lefty = int((-x * vy / vx) + y)
                righty = int(((lw - x) * vy / vx) + y)
                cv2.line(lane_roi_img, (lw - 1, righty), (0, lefty), (0, 255, 255), 3)
            if abs(vy) > 1e-5:
                steering_horizon_y = lh * 0.8
                target_x = int(((steering_horizon_y - y) * vx / vy) + x)
                if 0 < target_x < lw:
                    self.lane_center_x = target_x + lx
                    cv2.circle(original_screenshot, (self.lane_center_x, int(ly + steering_horizon_y)), 15,
                               (255, 0, 255), -1)

        if len(points) > 0 and inlier_mask is not None and len(inlier_mask) == len(points):
            for i, point in enumerate(points):
                p_tuple = tuple(point.astype(int))
                color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)
                cv2.circle(lane_roi_img, p_tuple, 3, color, -1)
        else:
            if tracked_blobs:
                for (objectID, centroid) in tracked_blobs.items():
                    cv2.circle(lane_roi_img, (centroid[0], centroid[1]), 5, (255, 165, 0), -1)

        output_image = original_screenshot.copy()
        output_image[ly:ly + lh, lx:lx + lw] = lane_roi_img

        return output_image, lane_roi_img, lane_roi_img, mask_cleaned

    def compute_steering_action(self):
        """
        Computes the steering control signal using the original PID controller logic,
        including error smoothing, filtered derivative, and anti-windup.
        """
        if self.lane_center_x is None:
            self.score_window.append(0)
            self.integral = 0
            self.previous_error = 0
            self.filtered_derivative = 0
            return 0.0

        error = self.lane_center_x - self.region_center_x
        self.error_window.append(error)
        smoothed_error = sum(self.error_window) / len(self.error_window)

        if len(self.error_window) > 0:
            current_score = 1000 / (1 + abs(smoothed_error))
            self.score_window.append(current_score)
        if len(self.score_window) > 0:
            self.stability_score = sum(self.score_window) / len(self.score_window)

        current_time = time.time()
        elapsed_time = current_time - self.previous_time
        if elapsed_time <= 0: return 0.0

        with self.lock:
            Kp, Ki, Kd, gain = self.Kp, self.Ki, self.Kd, self.steering_gain

        p_term = Kp * smoothed_error
        raw_derivative = (smoothed_error - self.previous_error) / elapsed_time
        alpha = elapsed_time / (self.derivative_filter_time_constant + elapsed_time)
        self.filtered_derivative = (1 - alpha) * self.filtered_derivative + alpha * raw_derivative
        d_term = Kd * self.filtered_derivative
        i_term = Ki * self.integral
        unclamped_signal = p_term + i_term + d_term
        final_control_signal = np.clip(unclamped_signal * gain / 100.0, -1.0, 1.0)
        windup_error = final_control_signal - (unclamped_signal * gain / 100.0)
        integral_adjustment = self.tracking_gain * windup_error
        self.integral += (smoothed_error * elapsed_time) - integral_adjustment
        self.previous_error = smoothed_error
        self.previous_time = current_time

        return final_control_signal


class CentroidTracker:
    """
    A simple object tracker based on centroid proximity. Preserved from the
    original script to maintain its behavior.
    """

    def __init__(self, maxDisappeared=10):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects


class SmoothedLine:
    """
    Represents a line with smoothed parameters over time. Preserved from the
    original script.
    """

    def __init__(self, smoothing_factor=0.15):
        self.line_params = None
        self.smoothing_factor = smoothing_factor

    def update(self, new_params, force_update=False):
        if new_params is None:
            return self.line_params
        current_params = np.array(new_params)
        if self.line_params is None or force_update:
            self.line_params = current_params
        else:
            self.line_params = (self.smoothing_factor * current_params) + \
                               ((1 - self.smoothing_factor) * self.line_params)
        return self.line_params


def pwm_steering_loop():
    """
    A dedicated thread that translates the continuous control signal into
    discrete keyboard presses using PWM. Preserved from the original script.
    """
    global control_signal, shutdown_signal, agent_running
    cycle_duration = 1.0 / PWM_FREQUENCY
    last_key_pressed = None

    while not shutdown_signal:
        if not agent_running:
            if last_key_pressed:
                try:
                    keyboard.release(last_key_pressed)
                except:
                    pass
                last_key_pressed = None
            try:
                keyboard.release('a')
            except:
                pass
            try:
                keyboard.release('d')
            except:
                pass
            time.sleep(0.1)
            continue

        with control_lock:
            current_signal = control_signal

        key_to_press, press_duration = None, 0
        if current_signal > 0.01:
            key_to_press, other_key = 'd', 'a'
            press_duration = cycle_duration * current_signal
        elif current_signal < -0.01:
            key_to_press, other_key = 'a', 'd'
            press_duration = cycle_duration * abs(current_signal)
        else:
            other_key = None
            try:
                keyboard.release('a')
            except:
                pass
            try:
                keyboard.release('d')
            except:
                pass

        if other_key:
            try:
                keyboard.release(other_key)
            except:
                pass

        release_duration = cycle_duration - press_duration
        if key_to_press:
            try:
                keyboard.press(key_to_press)
                last_key_pressed = key_to_press
                if press_duration > 0: time.sleep(press_duration)
                keyboard.release(key_to_press)
                if release_duration > 0: time.sleep(release_duration)
            except:
                pass
        else:
            last_key_pressed = None
            time.sleep(cycle_duration)

    print("INFO: Steering loop shutting down. Releasing keys.")
    for key in ['a', 'd', 'w']:
        try:
            keyboard.release(key)
        except:
            pass


def processing_loop(agent):
    """
    The main processing thread. It continuously captures the screen, runs the
    perception pipeline, computes steering actions, and manages video recording.
    """
    global last_processed_frame, agent_running, shutdown_signal, DIAGNOSTICS_ON, control_signal
    sct = mss()
    video_writer = None

    while not shutdown_signal:
        start_time = time.time()
        try:
            sct_img = np.array(sct.grab(agent.bounding_box))
            frame_bgr = cv2.cvtColor(sct_img, cv2.COLOR_BGRA2BGR)
            final_img, roi_img, warped_img, mask_img = agent._detect_lanes_dual_pipeline(frame_bgr)

            # --- Handle Video Recording (New Feature) ---
            if agent_running and VIDEO_RECORDING_ENABLED and video_writer is None:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"recording_{timestamp}.mp4"
                video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE,
                                               (agent.bounding_box['width'], agent.bounding_box['height']))
                if LOGGING_ENABLED: log.info(f"Starting new recording session: {os.path.abspath(filename)}")

            if video_writer is not None and (not agent_running or not VIDEO_RECORDING_ENABLED):
                if LOGGING_ENABLED: log.info("Recording session saved.")
                video_writer.release()
                video_writer = None

            # --- Agent Control Logic (Original) ---
            if agent_running:
                computed_signal = agent.compute_steering_action()
                with control_lock:
                    control_signal = computed_signal
                keyboard.press('w')
            else:
                # This logic differs slightly from original but is more robust:
                # it ensures recording stops if agent is paused, not just at the end.
                if video_writer is not None:
                    if LOGGING_ENABLED: log.info("Recording session saved due to agent pause.")
                    video_writer.release()
                    video_writer = None
                with control_lock:
                    control_signal = 0.0
                keyboard.release('w')

            # --- HUD and Diagnostics Overlay (Original) ---
            with agent.lock:
                hud_text = [
                    f"Kp: {agent.Kp:.4f} Ki: {agent.Ki:.4f} Kd: {agent.Kd:.4f}",
                    f"Score: {agent.stability_score:.0f} | Error: {agent.previous_error:.2f}",
                    f"Signal: {control_signal:.2f} | Gain: {agent.steering_gain:.1f}",
                    f"RANSAC Angle: {agent.current_ransac_angle:.1f} deg"
                ]
            for i, line in enumerate(hud_text):
                cv2.putText(final_img, line, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if DIAGNOSTICS_ON:
                h, w = agent.bounding_box['height'] // 2, agent.bounding_box['width'] // 2
                q1 = cv2.resize(final_img, (w, h));
                cv2.putText(q1, "Final Result", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                q2 = cv2.resize(warped_img, (w, h));
                cv2.putText(q2, "Bird's-Eye View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                q3 = cv2.resize(roi_img, (w, h));
                cv2.putText(q3, "Masked ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                q4 = cv2.cvtColor(cv2.resize(mask_img, (w, h)), cv2.COLOR_GRAY2BGR);
                cv2.putText(q4, "Color Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                top_row, bottom_row = np.hstack((q1, q2)), np.hstack((q3, q4))
                combined_image = np.vstack((top_row, bottom_row))
            else:
                combined_image = final_img

            if video_writer is not None:
                video_writer.write(
                    cv2.resize(combined_image, (agent.bounding_box['width'], agent.bounding_box['height'])))

            with frame_lock:
                last_processed_frame = combined_image.copy()

            elapsed = time.time() - start_time
            sleep_time = (1.0 / FRAME_RATE) - elapsed
            if sleep_time > 0: time.sleep(sleep_time)
        except Exception as e:
            if LOGGING_ENABLED: log.error(f"ERROR in processing_loop: {e}", exc_info=True)
            time.sleep(1)

    if video_writer is not None:
        video_writer.release()
        if LOGGING_ENABLED: log.info("Final recording session saved on shutdown.")


def run_genetic_tuner(agent):
    """
    Performs PID tuning using a Genetic Algorithm. Preserved from the
    original script.
    """
    global agent_running, shutdown_signal

    def test_individual(pid_settings):
        if shutdown_signal: return -1
        kp, ki, kd = pid_settings['kp'], pid_settings['ki'], pid_settings['kd']
        if LOGGING_ENABLED: log.info(f"GENETIC_TUNER: Testing -> Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}")
        agent.update_pid(kp, ki, kd)
        start_time = time.time()
        while time.time() - start_time < GENETIC_ALGORITHM_CONFIG["duration_per_test"]:
            if shutdown_signal: return -1
            time.sleep(0.5)
        avg_score = agent.stability_score
        if LOGGING_ENABLED: log.info(f"GENETIC_TUNER: Score = {avg_score:.2f}")
        return avg_score

    if LOGGING_ENABLED: log.info("\n--- STARTING GENETIC ALGORITHM PID TUNER ---")
    with ui_lock:
        ui_status["agent_status"] = "Genetic Tuning"
    agent_running = True
    time.sleep(0.1)

    cfg = GENETIC_ALGORITHM_CONFIG
    population = [{k: np.random.uniform(*cfg["search_space"][k]) for k in cfg["search_space"]} for _ in
                  range(cfg["population_size"])]
    best_ever_individual, best_ever_score = None, -1
    all_results = []

    try:
        for gen in range(cfg["num_generations"]):
            if shutdown_signal: break
            if LOGGING_ENABLED: log.info(f"\n--- GENERATION {gen + 1}/{cfg['num_generations']} ---")
            with ui_lock:
                ui_status["tune_iterations"] = gen + 1

            fitness_scores = [test_individual(ind) for ind in population]
            if -1 in fitness_scores: raise KeyboardInterrupt

            if max(fitness_scores) > best_ever_score:
                best_ever_score = max(fitness_scores)
                best_ever_individual = population[np.argmax(fitness_scores)]
                if LOGGING_ENABLED: log.info(
                    f"*** New Best Found! Score: {best_ever_score:.2f}, PID: {best_ever_individual} ***")
                with ui_lock:
                    ui_status["best_score"] = best_ever_score
                    ui_status["best_pid"] = best_ever_individual
                save_config()

            for i, score in enumerate(fitness_scores):
                res = {**population[i], 'StabilityScore': score, 'Generation': gen + 1}
                all_results.append(res)

            sorted_indices = np.argsort(fitness_scores)[::-1]
            parents = [population[i] for i in sorted_indices[:cfg["population_size"] // 2]]
            next_population = parents[:]
            while len(next_population) < cfg["population_size"]:
                p1, p2 = np.random.choice(parents, 2, replace=False)
                child = {k: (p1[k] + p2[k]) / 2 for k in p1}
                for param in child:
                    if np.random.rand() < cfg["mutation_rate"]:
                        child[param] *= (1 + np.random.uniform(-cfg["mutation_strength"], cfg["mutation_strength"]))
                        child[param] = np.clip(child[param], *cfg["search_space"][param])
                next_population.append(child)
            population = next_population
    except KeyboardInterrupt:
        print("\nGenetic Tuner interrupted by user.")
    finally:
        agent_running = False
        with ui_lock:
            ui_status["agent_status"] = "Paused"
        if all_results:
            output_file = 'genetic_tuning_results.csv'
            pd.DataFrame(all_results).to_csv(output_file, index=False)
            print(f"\n--- Genetic Tuning Complete ---")
            print(f"Results saved to {os.path.abspath(output_file)}")
            if best_ever_individual:
                print(f"Absolute Best Result: Score={best_ever_score:.2f}")
                print(
                    f" -> Kp={best_ever_individual['kp']:.4f}, Ki={best_ever_individual['ki']:.4f}, Kd={best_ever_individual['kd']:.4f}")
                agent.update_pid(**best_ever_individual)
                save_config()
        else:
            print("Tuning cancelled, no results saved.")


# ==============================================================================
# --- Web Server (Flask & Waitress) ---
# ==============================================================================
app = Flask(__name__, template_folder='.')
agent_instance = None


@app.route('/')
def index():
    """Serves the main HTML dashboard."""
    return render_template('dashboard.html')


@app.route('/status')
def status():
    """Provides a JSON object with the current state of the agent for the UI."""
    with agent_instance.lock, ui_lock, control_lock:
        data_to_send = {
            "agent_status": ui_status["agent_status"],
            "current_score": agent_instance.stability_score,
            "best_score": ui_status["best_score"],
            "best_ever_score": ui_status.get("best_ever_score", 0.0),
            "diagnostics_on": DIAGNOSTICS_ON,
            "video_recording": VIDEO_RECORDING_ENABLED,
            "data_logging": LOGGING_ENABLED,
            "tune_iterations": ui_status["tune_iterations"],
            "control_signal": control_signal,
            "current_error": agent_instance.previous_error,
            "pid": {"kp": agent_instance.Kp, "ki": agent_instance.Ki, "kd": agent_instance.Kd},
            "best_pid": ui_status["best_pid"],
            "gain": agent_instance.steering_gain,
            "tuning_mode": ui_status["tuning_mode"],
            "roi_ratios": config.get('roi_ratios')
        }
    return jsonify(data_to_send)


@app.route('/control', methods=['POST'])
def control():
    """Handles control commands sent from the web UI."""
    global agent_running, shutdown_signal, DIAGNOSTICS_ON, agent_instance
    global VIDEO_RECORDING_ENABLED, LOGGING_ENABLED
    data = request.json
    command, value = data.get('command'), data.get('value')

    if command == 'start_agent':
        if not agent_running:
            print("UI: Starting agent...")
            agent_running = True
            with ui_lock: ui_status["agent_status"] = "Running"
    elif command == 'pause_agent':
        if agent_running:
            print("UI: Pausing agent...")
            agent_running = False
            with ui_lock: ui_status["agent_status"] = "Paused"
    elif command == 'toggle_diagnostics':
        DIAGNOSTICS_ON = not DIAGNOSTICS_ON
        if DIAGNOSTICS_ON:
            log.addHandler(console_handler)
            print("\n--- CONSOLE LOGGING ENABLED ---")
        else:
            log.removeHandler(console_handler)
            print("\n--- CONSOLE LOGGING DISABLED ---")
    elif command == 'toggle_video_recording':
        VIDEO_RECORDING_ENABLED = not VIDEO_RECORDING_ENABLED
        with ui_lock:
            ui_status["video_recording"] = VIDEO_RECORDING_ENABLED
        print(f"UI: Video recording toggled {'ON' if VIDEO_RECORDING_ENABLED else 'OFF'}")
    elif command == 'toggle_data_logging':
        LOGGING_ENABLED = not LOGGING_ENABLED
        with ui_lock:
            ui_status["data_logging"] = LOGGING_ENABLED
        if LOGGING_ENABLED:
            log.addHandler(file_handler)
            if LOGGING_ENABLED: log.info("UI command: File logging ENABLED.")
        else:
            if LOGGING_ENABLED: log.info("UI command: File logging DISABLED.")
            log.removeHandler(file_handler)
        print(f"UI: Data logging toggled {'ON' if LOGGING_ENABLED else 'OFF'}")
    elif command == 'quit':
        print("UI: Quit signal received. Shutting down.")
        shutdown_signal = True
    elif command == 'set_mode':
        with ui_lock:
            ui_status["tuning_mode"] = value
        print(f"UI: Tuning mode set to {value}")
    elif command == 'set_kp':
        agent_instance.update_pid(value, agent_instance.Ki, agent_instance.Kd)
    elif command == 'set_ki':
        agent_instance.update_pid(agent_instance.Kp, value, agent_instance.Kd)
    elif command == 'set_kd':
        agent_instance.update_pid(agent_instance.Kp, agent_instance.Ki, value)
    elif command == 'set_gain':
        agent_instance.update_gain(value)
    return jsonify(success=True)


@app.route('/video_feed')
def video_feed():
    """Provides the real-time video stream to the web UI."""

    def stream_generator():
        while not shutdown_signal:
            with frame_lock:
                if last_processed_frame is None:
                    time.sleep(0.1)
                    continue
                frame = last_processed_frame
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ret: continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1.0 / FRAME_RATE)

    return Response(stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


def run_flask_app():
    """Runs the Flask web server using Waitress."""
    serve(app, host='0.0.0.0', port=5000, threads=8)


def get_local_ip():
    """Retrieves the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


# ==============================================================================
# --- Main Execution Block ---
# ==============================================================================
if __name__ == '__main__':
    config = load_config()
    game_bounding_box, car_roi_rect, lane_roi_rect = convert_ratios_to_pixels(
        config['roi_ratios']['game'],
        config['roi_ratios']['car'],
        config['roi_ratios']['lane']
    )
    agent_instance = SelfDrivingAgent(game_bounding_box, car_roi_rect, lane_roi_rect)
    agent_instance.update_pid(config['pid']['kp'], config['pid']['ki'], config['pid']['kd'])
    agent_instance.update_gain(config['gain'])

    with ui_lock:
        ui_status["best_pid"] = config['best_pid']
        ui_status["best_ever_score"] = config['best_ever_score']
        ui_status["best_score"] = 0.0

    flask_thread = Thread(target=run_flask_app)
    processing_thread = Thread(target=processing_loop, args=(agent_instance,))
    steering_thread = Thread(target=pwm_steering_loop)

    flask_thread.start()
    processing_thread.start()
    steering_thread.start()

    local_ip = get_local_ip()
    print("\n--- Dashboard is live! ---")
    print(f"On this computer: http://127.0.0.1:5000")
    print(f"On your tablet/other devices on the same network: http://{local_ip}:5000")
    print("--- Use the Web UI for all controls. ---")
    print("--- Press Ctrl+C in this terminal to quit. ---")

    try:
        tuner_thread = None
        while not shutdown_signal:
            with ui_lock:
                mode = ui_status["tuning_mode"]
            if mode == 'auto' and agent_running and (tuner_thread is None or not tuner_thread.is_alive()):
                tuner_thread = Thread(target=run_genetic_tuner, args=(agent_instance,))
                tuner_thread.start()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Initiating graceful shutdown.")
    finally:
        print("SHUTDOWN: Cleaning up resources. Please wait...")
        shutdown_signal = True
        save_config()
        if agent_running:
            agent_running = False
            for key in ['w', 'a', 's', 'd']:
                try:
                    keyboard.release(key)
                except:
                    pass

        if tuner_thread is not None and tuner_thread.is_alive(): tuner_thread.join(timeout=1.0)
        processing_thread.join(timeout=1.0)
        steering_thread.join(timeout=1.0)

        print("Program terminated successfully.")
        os._exit(0)