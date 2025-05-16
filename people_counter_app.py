# -*- coding: utf-8 -*-
import numpy as np
import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment as linear_assignment
import time
import mysql.connector
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIGURABLE SECTION: Update these as needed
# Fixed RTSP URL format (removed ? after password)
RTSP_STREAM = "rtsp://admin:D0youuseHIK?@192.168.3.66:554/Streaming/Channels/101"

# Alternative RTSP URLs to try if the main one fails
ALTERNATIVE_RTSP_URLS = [
    "rtsp://admin:D0youuseHIK?@192.168.3.66/Streaming/Channels/101",
    "rtsp://admin:D0youuseHIK%3F@192.168.3.66:554/h264/ch1/main/av_stream",
    "rtsp://192.168.3.66:554/Streaming/Channels/101",
    "rtsp://admin:D0youuseHIK?@192.168.3.66"
]

# MySQL database configuration
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",  # Update with your MySQL username
    "password": "password",  # Update with your MySQL password
    "database": "people_count"
}

# Detection parameters
CONFIDENCE_THRESHOLD = 0.4
MAX_AGE = 30
MIN_HITS = 3
IOU_THRESHOLD = 0.3

# Enable fallback mode if camera connection fails
ENABLE_FALLBACK = True

# Create database and tables if they don't exist
def setup_database():
    """Create necessary database and tables if they don't exist"""
    try:
        # Connect without specifying database to create it if needed
        conn = mysql.connector.connect(
            host=MYSQL_CONFIG["host"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"]
        )
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
        cursor.execute(f"USE {MYSQL_CONFIG['database']}")
        
        # Create main count table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people_count_main (
                id INT AUTO_INCREMENT PRIMARY KEY,
                current_count INT NOT NULL,
                people_in INT NOT NULL,
                people_out INT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        
        # Create transaction table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people_count_transactions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                event_type ENUM('IN', 'OUT') NOT NULL,
                previous_count INT NOT NULL,
                new_count INT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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

def insert_people_count(in_count, out_count, current_count):
    """Update the main count table with current totals"""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        
        # Update the main count table
        cursor.execute("""
            UPDATE people_count_main 
            SET current_count = %s, people_in = %s, people_out = %s, 
                last_updated = CURRENT_TIMESTAMP
            LIMIT 1
        """, (current_count, in_count, out_count))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"MySQL Update Error: {e}")
        return False

def add_transaction(event_type, previous_count, new_count, confidence=None):
    """Record a single counting transaction (in or out event)"""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO people_count_transactions 
            (event_type, previous_count, new_count, confidence)
            VALUES (%s, %s, %s, %s)
        """, (event_type, previous_count, new_count, confidence))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"MySQL Insert Error: {e}")
        return False

def get_current_counts():
    """Get the current counts from the database"""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT * FROM people_count_main LIMIT 1")
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        return result
    except Exception as e:
        logger.error(f"MySQL Select Error: {e}")
        return None

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
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
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
        self.confidence = None  # Store detection confidence

    def update(self, bbox, confidence=None):
        """
        Updates the state vector with observed bbox
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Store confidence if provided
        if confidence is not None:
            self.confidence = confidence
        
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.
        cy = (y1 + y2) / 2.
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1 + 1e-6)
        self.kf.update(np.array([cx, cy, s, r]).reshape((4,1)))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box
        """
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
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

def associate_detections_to_trackers(detections, trackers, iou_threshold=IOU_THRESHOLD):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
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
    SORT - Simple Online and Realtime Tracking
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

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns a similar array, where the last column is the object ID.
        """
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            bbox = trk.get_state()
            trks[t, :4] = bbox
            trks[t, 4] = 0
            if np.any(np.isnan(bbox)) or np.any(np.isinf(bbox)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets[:, :4], trks[:, :4], self.iou_threshold)
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4], confidence=dets[m[0], 4])

        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            trk.confidence = dets[i, 4] if dets.shape[1] > 4 else None
            self.trackers.append(trk)

        ret = []
        for i, trk in reversed(list(enumerate(self.trackers))):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Add tracking ID and confidence to output
                ret.append(np.concatenate((d, [trk.id + 1, trk.confidence if trk.confidence else 0])).reshape(1, -1))
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return np.concatenate(ret) if len(ret) > 0 else np.empty((0, 6))

def try_connect_camera(url, backend=None):
    """Try to connect to camera with specific URL and backend"""
    logger.info(f"Trying camera connection with URL: {url}")
    
    try:
        if backend:
            logger.info(f"Using backend: {backend}")
            cap = cv2.VideoCapture(url, backend)
        else:
            cap = cv2.VideoCapture(url)
        
        # Set additional properties for better streaming
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
        
        # Check if opened successfully
        if not cap.isOpened():
            logger.error(f"Failed to open camera with URL: {url}")
            return None
        
        # Try to read a test frame
        ret, frame = cap.read()
        if not ret:
            logger.error(f"Could open camera but failed to read frame from: {url}")
            cap.release()
            return None
        
        logger.info(f"Successfully connected to camera with URL: {url}")
        return cap
    except Exception as e:
        logger.error(f"Error connecting to camera with URL {url}: {e}")
        return None

def connect_to_camera():
    """Try multiple approaches to connect to the camera"""
    # Try the main URL first
    backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, None]
    
    # Try main URL with different backends
    for backend in backends:
        backend_name = "FFMPEG" if backend == cv2.CAP_FFMPEG else "GSTREAMER" if backend == cv2.CAP_GSTREAMER else "Default"
        logger.info(f"Trying main URL with {backend_name} backend")
        cap = try_connect_camera(RTSP_STREAM, backend)
        if cap:
            return cap
    
    # Try alternative URLs
    for url in ALTERNATIVE_RTSP_URLS:
        for backend in backends:
            cap = try_connect_camera(url, backend)
            if cap:
                return cap
    
    logger.error("Failed to connect to camera with all URLs and backends")
    return None

def main():
    # Make sure database is set up
    if not setup_database():
        logger.error("Failed to set up database. Exiting.")
        sys.exit(1)

    # Load current counts from database
    counts = get_current_counts()
    if counts:
        people_in = counts['people_in']
        people_out = counts['people_out']
        current_count = counts['current_count']
        logger.info(f"Loaded counts from DB: In={people_in}, Out={people_out}, Current={current_count}")
    else:
        people_in = 0
        people_out = 0
        current_count = 0
    
    # Load YOLOv8 model
    logger.info("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    
    # Initialize tracker
    tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)
    
    # Connect to camera
    logger.info("Connecting to camera...")
    cap = connect_to_camera()
    
    if not cap and not ENABLE_FALLBACK:
        logger.error("Failed to connect to camera and fallback mode is disabled. Exiting.")
        sys.exit(1)
    
    # Create a blank fallback frame if needed
    fallback_frame = None
    using_fallback = False
    if not cap and ENABLE_FALLBACK:
        logger.warning("Using fallback mode with static frame")
        using_fallback = True
        fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            fallback_frame, 
            "Camera connection failed", 
            (50, 200), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
        cv2.putText(
            fallback_frame, 
            f"In: {people_in} | Out: {people_out} | Current: {current_count}", 
            (50, 240), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
    
    # Initialize frame dimensions and counting line
    if cap and not using_fallback:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read initial frame from camera")
            if ENABLE_FALLBACK:
                logger.warning("Switching to fallback mode")
                using_fallback = True
                frame = fallback_frame
            else:
                logger.error("Exiting due to camera error")
                cap.release()
                sys.exit(1)
        
        # Set counting line in middle of frame
        frame_width = frame.shape[1]
        line_x = frame_width // 2
    else:
        frame_width = 640  # Default for fallback mode
        line_x = frame_width // 2
    
    # Initialize tracking memory
    track_memory = {}
    reconnect_attempts = 0
    last_reconnect_time = 0
    
    logger.info("Starting main loop...")
    while True:
        # Handle camera reconnection if needed
        if not cap and not using_fallback:
            logger.error("Camera connection lost")
            if ENABLE_FALLBACK:
                logger.warning("Switching to fallback mode")
                using_fallback = True
            else:
                logger.error("Exiting due to camera error")
                break
        
        if using_fallback and time.time() - last_reconnect_time > 30:  # Try reconnecting every 30 seconds
            logger.info(f"Fallback mode active. Reconnection attempt {reconnect_attempts+1}")
            last_reconnect_time = time.time()
            reconnect_attempts += 1
            
            cap = connect_to_camera()
            if cap:
                logger.info("Successfully reconnected to camera")
                using_fallback = False
                reconnect_attempts = 0
                
                # Update frame width and line position
                ret, frame = cap.read()
                if ret:
                    frame_width = frame.shape[1]
                    line_x = frame_width // 2
        
        # Get frame from camera or use fallback
        if not using_fallback:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                if ENABLE_FALLBACK:
                    logger.warning("Switching to fallback mode temporarily")
                    using_fallback = True
                    continue
                else:
                    break
        else:
            # Update fallback frame with current counts
            fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                fallback_frame, 
                "Camera connection failed - FALLBACK MODE", 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
            cv2.putText(
                fallback_frame, 
                f"In: {people_in} | Out: {people_out} | Current: {current_count}", 
                (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            frame = fallback_frame
        
        # Skip processing in fallback mode
        if using_fallback:
            cv2.imshow('People Counting', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.1)  # Slow down loop in fallback mode
            continue
        
        # Run YOLOv8 detection
        results = model(frame)[0]
        
        # Extract person detections
        dets = []
        for r in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, cls = r
            if int(cls) == 0 and score > CONFIDENCE_THRESHOLD:  # Class 0 is person
                dets.append([x1, y1, x2, y2, score])
        dets = np.array(dets)
        
        # Update tracker
        if len(dets) > 0:
            tracks = tracker.update(dets)
        else:
            tracks = tracker.update(np.empty((0, 5)))
        
        # Process tracking results
        current_centroids = {}
        
        for d in tracks:
            x1, y1, x2, y2, track_id, confidence = d
            track_id = int(track_id)
            
            # Calculate centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Draw bounding box and ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Check previous position
            prev_cx = track_memory.get(track_id, None)
            
            # Check if person crossed the counting line
            if prev_cx is not None:
                previous_count = current_count
                
                # Crossed from left to right
                if prev_cx < line_x and cx >= line_x:
                    people_in += 1
                    current_count += 1
                    
                    # Update database
                    insert_people_count(people_in, people_out, current_count)
                    add_transaction('IN', previous_count, current_count, confidence)
                    
                    logger.info(f"Person entered (ID: {track_id}). Current count: {current_count}")
                
                # Crossed from right to left
                elif prev_cx >= line_x and cx < line_x:
                    people_out += 1
                    current_count = max(0, current_count - 1)  # Ensure not negative
                    
                    # Update database
                    insert_people_count(people_in, people_out, current_count)
                    add_transaction('OUT', previous_count, current_count, confidence)
                    
                    logger.info(f"Person exited (ID: {track_id}). Current count: {current_count}")
            
            # Update track memory
            current_centroids[track_id] = cx
        
        # Update track memory
        track_memory = current_centroids
        
        # Draw counting line
        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 2)
        
        # Draw counts
        cv2.putText(frame, f'In: {people_in}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Out: {people_out}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Current: {current_count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('People Counting', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    if cap and not using_fallback:
        cap.release()
    cv2.destroyAllWindows()
    logger.info("Application stopped")

if __name__ == "__main__":
    main()
