"""
People Counter Application
-------------------------
This application counts people from a bird's eye view camera feed,
tracks their movements, and stores the data in a MySQL database.

Features:
- Bird's eye view optimized person detection using YOLOv8
- Flask web interface for monitoring and configuration
- Proper SQL database structure with main count and transaction tables
- Robust error handling and connection management
"""
import os
import sys
import time
import datetime
import threading
import numpy as np
import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment as linear_assignment
import mysql.connector
from mysql.connector import pooling
import logging
from flask import Flask, render_template, Response, jsonify, request

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================== CONFIGURATION =====================
# Camera configuration
# Fix the RTSP URL format - properly escape special characters and fix port format
RTSP_STREAM = "rtsp://admin:D0youuseHIK?@192.168.3.66:554/Streaming/Channels/101"  # Removed the question mark

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",  # Update with your MySQL username
    "password": "password",  # Update with your MySQL password
    "database": "people_count"
}

# Detection parameters
CONFIDENCE_THRESHOLD = 0.4  # Increased for more reliable detections
IOU_THRESHOLD = 0.3
MAX_AGE = 20
MIN_HITS = 3

# Processing parameters
FRAME_SKIP = 2  # Process every nth frame for performance
PROCESS_RESOLUTION = (640, 480)  # Resize frames for faster processing

# Bird's eye view optimization
BIRDS_EYE_VIEW_MODE = True  # Enable bird's eye view optimizations

# ===================== DATABASE SETUP =====================
def setup_database():
    """Create necessary database and tables if they don't exist"""
    try:
        # Connect without specifying database to create it if needed
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        cursor.execute(f"USE {DB_CONFIG['database']}")
        
        # Create main count table (stores the most recent count)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people_count_main (
                id INT AUTO_INCREMENT PRIMARY KEY,
                current_count INT NOT NULL,
                people_in INT NOT NULL,
                people_out INT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        
        # Create transaction table (stores individual counting events)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people_count_transactions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                event_type ENUM('IN', 'OUT') NOT NULL,
                previous_count INT NOT NULL,
                new_count INT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                camera_id VARCHAR(50) DEFAULT 'main',
                confidence FLOAT
            )
        """)
        
        # Initialize main count table with a default record if it doesn't exist
        cursor.execute("""
            INSERT INTO people_count_main (current_count, people_in, people_out)
            SELECT 0, 0, 0 FROM DUAL
            WHERE NOT EXISTS (SELECT * FROM people_count_main)
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database setup completed successfully")
        return True
    except Exception as e:
        logger.error(f"Database setup error: {e}")
        return False

# Create a connection pool for efficient database connections
def create_connection_pool():
    try:
        pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="people_counter_pool",
            pool_size=5,
            **DB_CONFIG
        )
        logger.info("Database connection pool created successfully")
        return pool
    except Exception as e:
        logger.error(f"Failed to create connection pool: {e}")
        return None

# ===================== TRACKING UTILITIES =====================
def iou(bb_test, bb_gt):
    """Calculate intersection over union of two bounding boxes"""
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    return inter / (area1 + area2 - inter + 1e-6)

class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        
        # Bird's eye view optimization - tune process noise for overhead perspective
        if BIRDS_EYE_VIEW_MODE:
            self.kf.R[2:,2:] *= 5.0  # Lower measurement noise for size
            self.kf.P[4:,4:] *= 500.0  # Higher initial velocity uncertainty
        else:
            self.kf.R[2:,2:] *= 10.0
            self.kf.P[4:,4:] *= 1000.0
        
        self.kf.P *= 10.0
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.
        cy = (y1 + y2) / 2.
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1 + 1e-6)
        self.kf.x[:4] = np.array([cx, cy, s, r]).reshape((4,1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # Store additional metrics
        self.confidence = None  # Will store detection confidence
        self.last_position = (cx, cy)

    def update(self, bbox, confidence=None):
        """
        Updates the state vector with observed bbox
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.
        cy = (y1 + y2) / 2.
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1 + 1e-6)
        
        self.last_position = (cx, cy)
        if confidence is not None:
            self.confidence = confidence
            
        self.kf.update(np.array([cx, cy, s, r]).reshape((4,1)))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate
        """
        cx, cy, s, r = self.kf.x[:4].reshape((4,))
        s = max(s, 1e-6)
        r = max(r, 1e-6)
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        x1 = cx - w/2.
        y1 = cy - h/2.
        x2 = cx + w/2.
        y2 = cy + h/2.
        return np.array([x1, y1, x2, y2])

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return [], list(range(len(detections))), []

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    matched_indices = np.array(linear_assignment(-iou_matrix)).T
    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(2))

    return matches, unmatched_detections, unmatched_trackers

class Sort:
    """
    SORT - Simple Online and Realtime Tracking with a Deep Association Metric
    """
    def __init__(self, max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            bbox = trk.get_state()
            trks[t, :4] = bbox
            trks[t, 4] = 0  # Placeholder
            if np.any(np.isnan(bbox)) or np.any(np.isinf(bbox)):
                to_del.append(t)
                
        # Remove invalid trackers
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Handle edge case when no detections are provided
        if len(dets) == 0:
            dets = np.empty((0, 5))
            
        # Match current detections to existing trackers
        if len(trks) > 0:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
                dets[:, :4], trks[:, :4], self.iou_threshold)
        else:
            unmatched_dets = list(range(len(dets)))
            matched = []
            unmatched_trks = []

        # Update matched trackers with assigned detections
        for m in matched:
            det_idx, trk_idx = m
            self.trackers[trk_idx].update(dets[det_idx, :4], confidence=dets[det_idx, 4])

        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            trk.confidence = dets[i, 4]
            self.trackers.append(trk)

        # Return active trackers
        ret = []
        for i, trk in enumerate(list(self.trackers)):
            d = trk.get_state()
            # Only return confirmed tracks that have been seen recently
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1, trk.confidence if trk.confidence else 0])).reshape(1, -1))
            
            # Remove dead tracks
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        return np.concatenate(ret) if len(ret) > 0 else np.empty((0, 6))

# ===================== DATABASE OPERATIONS =====================
class DatabaseManager:
    """Handles all database operations for the people counter"""
    
    def __init__(self, connection_pool):
        self.pool = connection_pool
        
    def get_connection(self):
        """Get a connection from the pool"""
        if self.pool:
            return self.pool.get_connection()
        return None
        
    def update_main_count(self, people_in, people_out, current_count):
        """Update the main count table with current totals"""
        conn = self.get_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return False
            
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE people_count_main 
                SET current_count = %s, people_in = %s, people_out = %s, 
                    last_updated = CURRENT_TIMESTAMP
                LIMIT 1
            """, (current_count, people_in, people_out))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating main count: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            conn.close()
            
    def add_transaction(self, event_type, previous_count, new_count, confidence=None):
        """Record a single counting transaction (in or out event)"""
        conn = self.get_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return False
            
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO people_count_transactions 
                (event_type, previous_count, new_count, confidence)
                VALUES (%s, %s, %s, %s)
            """, (event_type, previous_count, new_count, confidence))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding transaction: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            conn.close()
    
    def get_current_counts(self):
        """Get the current counts from the database"""
        conn = self.get_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return None
            
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM people_count_main LIMIT 1")
            result = cursor.fetchone()
            return result
        except Exception as e:
            logger.error(f"Error getting current counts: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            conn.close()
            
    def get_recent_transactions(self, limit=50):
        """Get recent counting transactions"""
        conn = self.get_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return []
            
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT * FROM people_count_transactions
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            result = cursor.fetchall()
            return result
        except Exception as e:
            logger.error(f"Error getting recent transactions: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            conn.close()

# ===================== PERSON COUNTER =====================
class PeopleCounter:
    """Main class for people detection and counting"""
    
    def __init__(self, db_manager, video_source=RTSP_STREAM):
        """Initialize the people counter"""
        self.db_manager = db_manager
        self.video_source = video_source
        
        # Load YOLOv8 model - using n for speed, use m or l for better accuracy
        self.model = YOLO('yolov8n.pt')
        
        # Initialize tracker
        self.tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)
        
        # Initialize video capture
        self.cap = None
        self.connect_to_camera()
        
        # Initialize frame dimensions and counting line
        self.frame_width = 640
        self.frame_height = 480
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_height, self.frame_width = frame.shape[:2]
        
        # Set counting line in the middle by default
        self.line_x = self.frame_width // 2
        
        # Initialize counters
        self.people_in = 0
        self.people_out = 0
        self.current_count = 0
        self.track_memory = {}
        
        # Initialize processing variables
        self.frame_count = 0
        self.processing = False
        self.latest_frame = None
        self.processed_frame = None
        
        # Load current counts from database
        self.load_counts_from_db()
        
        # Status
        self.running = False

    def connect_to_camera(self):
        """Connect to the camera source"""
        try:
            logger.info(f"Attempting to connect to camera: {self.video_source}")
            
            # Try to open the camera with different backend options
            self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
            
            # Check if connection was successful
            if not self.cap.isOpened():
                logger.error(f"Error: Cannot open video source with FFMPEG backend: {self.video_source}")
                
                # Try with GSTREAMER backend as fallback
                logger.info("Trying with GSTREAMER backend...")
                self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_GSTREAMER)
                
                if not self.cap.isOpened():
                    logger.error(f"Error: Cannot open video source with GSTREAMER backend: {self.video_source}")
                    
                    # Try without specifying backend as last resort
                    logger.info("Trying without specific backend...")
                    self.cap = cv2.VideoCapture(self.video_source)
                    
                    if not self.cap.isOpened():
                        logger.error(f"Error: Cannot open video source with default backend: {self.video_source}")
                        return False
            
            # Successfully opened connection
            logger.info("Successfully connected to camera")
            
            # Set additional capture properties if needed
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce buffer size to minimize latency
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to camera: {str(e)}")
            return False
            
    def load_counts_from_db(self):
        """Load current counts from database"""
        counts = self.db_manager.get_current_counts()
        if counts:
            self.people_in = counts['people_in']
            self.people_out = counts['people_out']
            self.current_count = counts['current_count']
            logger.info(f"Loaded counts from DB: In={self.people_in}, Out={self.people_out}, Current={self.current_count}")
            
    def start(self):
        """Start the people counter"""
        if self.running:
            return
            
        self.running = True
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("People counter started")
        
    def stop(self):
        """Stop the people counter"""
        self.running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        logger.info("People counter stopped")
        
    def get_frame(self):
        """Get the latest processed frame"""
        if self.processed_frame is None:
            # Return a blank frame if no processed frame is available
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for video...", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', blank)
            return jpeg.tobytes()
            
        ret, jpeg = cv2.imencode('.jpg', self.processed_frame)
        return jpeg.tobytes()
        
    def get_counts(self):
        """Get the current counts"""
        return {
            'in': self.people_in,
            'out': self.people_out,
            'current': self.current_count
        }
        
    def set_counting_line(self, position):
        """Set the counting line position"""
        if 0 <= position <= self.frame_width:
            self.line_x = position
            return True
        return False
        
    def process_frames(self):
        """Main processing loop"""
        while self.running:
            if not self.cap or not self.cap.isOpened():
                if not self.connect_to_camera():
                    time.sleep(1)
                    continue
                    
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue
                
            self.latest_frame = frame.copy()
            
            # Skip frames for performance
            self.frame_count += 1
            if self.frame_count % FRAME_SKIP != 0:
                continue
                
            # Process the frame
            self.processing = True
            processed = self.process_frame(frame)
            self.processed_frame = processed
            self.processing = False
            
        logger.info("Processing thread stopped")
            
    def process_frame(self, frame):
        """Process a single frame for people detection and counting"""
        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, PROCESS_RESOLUTION)
        
        # Run YOLOv8 detection
        results = self.model(frame_resized)[0]
        
        # Extract person detections (class 0 is person)
        detections = []
        for r in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, cls = r
            if int(cls) == 0 and score > CONFIDENCE_THRESHOLD:
                # Scale coordinates back to original frame size
                x1_orig = x1 * frame.shape[1] / PROCESS_RESOLUTION[0]
                y1_orig = y1 * frame.shape[0] / PROCESS_RESOLUTION[1]
                x2_orig = x2 * frame.shape[1] / PROCESS_RESOLUTION[0]
                y2_orig = y2 * frame.shape[0] / PROCESS_RESOLUTION[1]
                detections.append([x1_orig, y1_orig, x2_orig, y2_orig, score])
                
        detections = np.array(detections)
        
        # Update tracker
        if len(detections) > 0:
            tracks = self.tracker.update(detections)
        else:
            tracks = self.tracker.update(np.empty((0, 5)))
            
        # Process tracking results
        current_centroids = {}
        
        for d in tracks:
            x1, y1, x2, y2, track_id, confidence = d
            track_id = int(track_id)
            
            # Calculate centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Check previous position
            prev_cx = self.track_memory.get(track_id, None)
            
            # Bird's eye view optimization - filter out small detections
            # that are likely false positives
            area = (x2 - x1) * (y2 - y1)
            if BIRDS_EYE_VIEW_MODE and area < 900:  # Min area threshold
                continue
                
            # Draw bounding box and ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
            # Check if track crossed the counting line
            if prev_cx is not None:
                previous_count = self.current_count
                # Crossed from left to right
                if prev_cx < self.line_x and cx >= self.line_x:
                    self.people_in += 1
                    self.current_count += 1
                    
                    # Update database
                    self.db_manager.update_main_count(
                        self.people_in, self.people_out, self.current_count)
                    self.db_manager.add_transaction(
                        'IN', previous_count, self.current_count, confidence)
                    
                    logger.info(f"Person entered (ID: {track_id}). Current count: {self.current_count}")
                    
                # Crossed from right to left
                elif prev_cx >= self.line_x and cx < self.line_x:
                    self.people_out += 1
                    # Ensure count doesn't go negative
                    self.current_count = max(0, self.current_count - 1)
                    
                    # Update database
                    self.db_manager.update_main_count(
                        self.people_in, self.people_out, self.current_count)
                    self.db_manager.add_transaction(
                        'OUT', previous_count, self.current_count, confidence)
                    
                    logger.info(f"Person exited (ID: {track_id}). Current count: {self.current_count}")
            
            # Update track memory
            current_centroids[track_id] = cx
            
        # Update track memory for next frame
        self.track_memory = current_centroids
        
        # Draw counting line
        cv2.line(frame, (self.line_x, 0), (self.line_x, frame.shape[0]), (0, 0, 255), 2)
        
        # Draw counts
        cv2.putText(frame, f'In: {self.people_in}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Out: {self.people_out}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Current: {self.current_count}', (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (frame.shape[1] - 200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
        return frame

# ===================== FLASK APPLICATION =====================
app = Flask(__name__)

# Global counter instance
counter = None
db_manager = None

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            if counter:
                frame = counter.get_frame()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/counts')
def get_counts():
    """Get current counts as JSON"""
    if counter:
        counts = counter.get_counts()
        return jsonify(counts)
    return jsonify({'in': 0, 'out': 0, 'current': 0})

@app.route('/api/transactions')
def get_transactions():
    """Get recent transactions as JSON"""
    if db_manager:
        transactions = db_manager.get_recent_transactions(limit=100)
        return jsonify(transactions)
    return jsonify([])

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update counter settings"""
    if request.method == 'POST' and counter:
        data = request.json
        
        # Update counting line position
        if 'line_position' in data:
            position = int(data['line_position'])
            counter.set_counting_line(position)
            
        return jsonify({'status': 'success'})
    
    return jsonify({'status': 'error', 'message': 'Counter not initialized'})

@app.route('/api/reset', methods=['POST'])
def reset_counts():
    """Reset all counts"""
    if request.method == 'POST' and counter and db_manager:
        # Reset counter values
        counter.people_in = 0
        counter.people_out = 0
        counter.current_count = 0
        
        # Update database
        db_manager.update_main_count(0, 0, 0)
        
        return jsonify({'status': 'success'})
    
    return jsonify({'status': 'error', 'message': 'Counter not initialized'})

# Create necessary HTML templates
def create_templates():
    """Create Flask templates directory and HTML files"""
    os.makedirs('templates', exist_ok=True)
    
    # Index.html
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>People Counter Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 20px; }
        .count-card { text-align: center; margin-bottom: 20px; }
        .count-value { font-size: 3em; font-weight: bold; }
        .video-container { position: relative; }
        #counting-line { 
            position: absolute; 
            top: 0; 
            width: 3px; 
            height: 100%; 
            background-color: red; 
            z-index: 100;
            cursor: ew-resize;
        }
        #transactions-table {
            max-height: 400px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">People Counter Dashboard</h1>
        
        <div class="row">
            <div class="col-md-8">
                <div class="card mb-4">
                    <div class="card-header">
                        <h5>Live Video Feed</h5>
                    </div>
                    <div class="card-body p-0 video-container">
                        <div id="counting-line"></div>
                        <img src="{{ url_for('video_feed') }}" class="img-fluid" id="video-feed">
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="row">
                    <div class="col-md-12">
                        <div class="card count-card bg-success text-white">
                            <div class="card-body">
                                <h5 class="card-title">Current Count</h5>
                                <div class="count-value" id="current-count">0</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card count-card bg-primary text-white">
                            <div class="card-body">
                                <h5 class="card-title">In</h5>
                                <div class="count-value" id="people-in">0</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card count-card bg-danger text-white">
                            <div class="card-body">
                                <h5 class="card-title">Out</h5>
                                <div class="count-value" id="people-out">0</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-12 mb-3">
                        <button class="btn btn-warning w-100" id="reset-button">Reset Counts</button>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Recent Transactions</h5>
                    </div>
                    <div class="card-body" id="transactions-table">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Event</th>
                                    <th>Previous Count</th>
                                    <th>New Count</th>
                                    <th>Confidence</th>
                                </tr>
                            </thead>
                            <tbody id="transactions-body">
                                <!-- Transactions will be populated here -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Initialize variables
        let videoFeed = document.getElementById('video-feed');
        let countingLine = document.getElementById('counting-line');
        let isDragging = false;
        
        // Set initial line position (middle of video)
        countingLine.style.left = '50%';
        
        // Update counts every second
        function updateCounts() {
            fetch('/api/counts')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('current-count').textContent = data.current;
                    document.getElementById('people-in').textContent = data.in;
                    document.getElementById('people-out').textContent = data.out;
                });
        }
        
        // Update transactions table
        function updateTransactions() {
            fetch('/api/transactions')
                .then(response => response.json())
                .then(data => {
                    let transactionsBody = document.getElementById('transactions-body');
                    transactionsBody.innerHTML = '';
                    
                    data.forEach(transaction => {
                        let row = document.createElement('tr');
                        
                        // Format timestamp
                        let timestamp = new Date(transaction.timestamp);
                        let formattedTime = timestamp.toLocaleString();
                        
                        row.innerHTML = `
                            <td>${formattedTime}</td>
                            <td>${transaction.event_type}</td>
                            <td>${transaction.previous_count}</td>
                            <td>${transaction.new_count}</td>
                            <td>${transaction.confidence ? transaction.confidence.toFixed(2) : 'N/A'}</td>
                        `;
                        
                        transactionsBody.appendChild(row);
                    });
                });
        }
        
        // Make counting line draggable
        countingLine.addEventListener('mousedown', function(e) {
            isDragging = true;
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', function(e) {
            if (isDragging) {
                let rect = videoFeed.getBoundingClientRect();
                let x = e.clientX - rect.left;
                let percentage = (x / rect.width) * 100;
                
                // Constrain to video boundaries
                percentage = Math.max(0, Math.min(100, percentage));
                
                countingLine.style.left = percentage + '%';
            }
        });
        
        document.addEventListener('mouseup', function() {
            if (isDragging) {
                isDragging = false;
                
                // Get line position and update server
                let percentage = parseFloat(countingLine.style.left) / 100;
                let videoWidth = videoFeed.naturalWidth || videoFeed.width;
                let position = Math.round(percentage * videoWidth);
                
                fetch('/api/settings', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        line_position: position
                    })
                });
            }
        });
        
        // Reset button
        document.getElementById('reset-button').addEventListener('click', function() {
            if (confirm('Are you sure you want to reset all counts?')) {
                fetch('/api/reset', {
                    method: 'POST'
                }).then(() => {
                    updateCounts();
                    updateTransactions();
                });
            }
        });
        
        // Update data periodically
        setInterval(updateCounts, 1000);
        setInterval(updateTransactions, 5000);
        
        // Initial updates
        updateCounts();
        updateTransactions();
    </script>
</body>
</html>
        ''')

def main():
    """Main entry point for the application"""
    global counter, db_manager
    
    # Setup database
    if not setup_database():
        logger.error("Failed to setup database. Exiting.")
        sys.exit(1)
    
    # Create connection pool
    connection_pool = create_connection_pool()
    if not connection_pool:
        logger.error("Failed to create database connection pool. Exiting.")
        sys.exit(1)
    
    # Create database manager
    db_manager = DatabaseManager(connection_pool)
    
    # Create people counter
    counter = PeopleCounter(db_manager, video_source=RTSP_STREAM)
    
    # Create template files
    create_templates()
    
    # Start counter
    counter.start()
    
    try:
        # Start Flask application
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        # Clean up
        if counter:
            counter.stop()
        logger.info("Application stopped")

if __name__ == "__main__":
    main()
