# -*- coding: utf-8 -*-

import numpy as np
import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment as linear_assignment
import mysql.connector
from mysql.connector import Error

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    o = inter / (area1 + area2 - inter + 1e-6)
    return o

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
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

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.
        cy = (y1 + y2) / 2.
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1 + 1e-6)

        self.kf.update(np.array([cx, cy, s, r]).reshape((4,1)))

    def predict(self):
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
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
    if len(trackers) == 0:
        return [], list(range(len(detections))), []

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    matched_indices = np.array(linear_assignment(-iou_matrix)).T

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(2))

    return matches, unmatched_detections, unmatched_trackers

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            bbox = trk.get_state()
            trks[t, :4] = bbox
            trks[t, 4] = 0
            if np.any(np.isnan(bbox)) or np.any(np.isinf(bbox)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)

        i = len(self.trackers)
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

# MySQL Connection Setup
def connect_db():
    try:
        conn = mysql.connector.connect(
            host='localhost',        # Change to your DB host
            database='people_count', # Change to your DB name
            user='vara',     # Change to your DB user
            password='BhSU1046!'  # Change to your DB password
        )
        if conn.is_connected():
            print("Connected to MySQL database")
            return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def insert_counts(conn, people_in, people_out, current_count):
    try:
        cursor = conn.cursor()
        sql = """
            INSERT INTO people_count (people_in, people_out, current_count)
            VALUES (%s, %s, %s)
        """
        cursor.execute(sql, (people_in, people_out, current_count))
        conn.commit()
        cursor.close()
    except Error as e:
        print(f"Error inserting counts into DB: {e}")

def main(video_source=0):
    conn = connect_db()
    if conn is None:
        print("Database connection failed. Exiting.")
        return

    model = YOLO('yolov8n.pt')
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {video_source}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to read from video source")
        return
    line_x = frame.shape[1] // 2

    people_in = 0
    people_out = 0
    current_count = 0

    track_memory = {}

    last_people_in = people_in
    last_people_out = people_out
    last_current_count = current_count

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        dets = []
        for r in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, cls = r
            if int(cls) == 0 and score > 0.3:
                dets.append([x1, y1, x2, y2, score])
        dets = np.array(dets)

        tracks = tracker.update(dets)
        current_centroids = {}

        for d in tracks:
            x1, y1, x2, y2, track_id = d
            cx = int((x1 + x2) / 2)

            prev_cx = track_memory.get(track_id, None)

            if prev_cx is not None:
                if prev_cx < line_x <= cx:
                    people_in += 1
                    current_count += 1
                elif prev_cx > line_x >= cx:
                    people_out += 1
                    current_count = max(0, current_count - 1)

            current_centroids[track_id] = cx

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {int(track_id)}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        track_memory = current_centroids
        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 2)

        cv2.putText(frame, f'In: {people_in}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Out: {people_out}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Current: {current_count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Insert into DB only if counts changed
        if (people_in != last_people_in) or (people_out != last_people_out) or (current_count != last_current_count):
            insert_counts(conn, people_in, people_out, current_count)
            last_people_in = people_in
            last_people_out = people_out
            last_current_count = current_count

        cv2.imshow('People Counting', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main(0)
