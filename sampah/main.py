import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import math
from flask import Flask, Response, render_template

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# --- KONFIGURASI GLOBAL ---
YOLO_MODEL_PATH = 'yolov8n.pt'
PROCESSING_WIDTH = 480
JPEG_QUALITY = 75
VEHICLE_CLASSES = [2, 3, 5, 7] # mobil, motor, bis, truk

LINE_COLOR = (0, 255, 255)
TEXT_COLOR = (0, 0, 255)
BOX_COLOR = (0, 255, 0)
WARNING_COLOR = (0, 165, 255)
ACCIDENT_COLOR = (0, 0, 255)
# --- AKHIR KONFIGURASI GLOBAL ---

# --- PARAMETER PER VIDEO ---
# Di sini Anda bisa mengatur parameter yang berbeda untuk setiap video
VIDEO_SPECIFIC_PARAMS = {
    'crash': {
        'MIN_CONFIDENCE': 0.15, # Kepercayaan minimum deteksi YOLO
        'MIN_TRACK_FRAMES': 5,  # Frame minimum sebuah track dianggap valid
        'SMOOTHING_HISTORY': 30,# Jumlah frame untuk smoothing bounding box
        'ACCIDENT_THRESHOLD': 7, # Frame abnormal berturut-turut untuk deteksi kecelakaan
        'COLLISION_IOU_THRESH': 0.1, # IoU threshold untuk deteksi tabrakan
        'MAX_MISSED_FRAMES': 10, # Frame maksimum sebuah objek bisa hilang sebelum track dihapus
        'AR_CHANGE_THRESH': 1.8, # Perubahan aspek rasio untuk deteksi abnormal
        'ANGLE_CHANGE_THRESH': 120, # Perubahan sudut untuk deteksi abnormal (derajat)
        'SPEED_CHANGE_THRESH': 15, # Perubahan kecepatan untuk deteksi abnormal (pixel/frame)
        'MIN_MOVEMENT': 5,       # Gerakan minimum (pixel) untuk dianggap bergerak
    },
    'highway': {
        'MIN_CONFIDENCE': 0.25, # Contoh: Mungkin video jalan raya butuh confidence lebih tinggi
        'MIN_TRACK_FRAMES': 3,
        'SMOOTHING_HISTORY': 25, # Mungkin smoothing lebih sedikit agar lebih responsif
        'ACCIDENT_THRESHOLD': 10,
        'COLLISION_IOU_THRESH': 0.12,
        'MAX_MISSED_FRAMES': 15,
        'AR_CHANGE_THRESH': 2.0,
        'ANGLE_CHANGE_THRESH': 100,
        'SPEED_CHANGE_THRESH': 20,
        'MIN_MOVEMENT': 7,
    }
}
# --- AKHIR PARAMETER PER VIDEO ---

# Muat Model YOLO
try:
    print(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# --- Fungsi Helper (Sama seperti sebelumnya, tidak diubah) ---
def smooth_boxes(cur_box, prev_box_list, alpha=0.6):
    if not prev_box_list: return cur_box
    weights = np.linspace(0.2, 1.0, len(prev_box_list))
    weights /= weights.sum()
    weighted_sum = np.zeros_like(cur_box, dtype=float)
    for box, weight in zip(prev_box_list, weights):
        weighted_sum += np.array(box, dtype=float) * weight
    return alpha * np.array(cur_box, dtype=float) + (1 - alpha) * weighted_sum

def aspect_ratio(box):
    x1, y1, x2, y2 = box
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    return min(max(w / h if h > 0 else w, 0.1), 10)

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0]); y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2]); y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0

# Modifikasi: check_abnormal_movement sekarang menerima `params`
def check_abnormal_movement(tid, current_box, prev_boxes_list, params):
    if len(prev_boxes_list) < 3: return False
    boxes_to_analyze = list(prev_boxes_list)[-3:] + [current_box]
    ar_changes = []
    for i in range(1, len(boxes_to_analyze)):
        ar_prev = aspect_ratio(boxes_to_analyze[i-1]); ar_curr = aspect_ratio(boxes_to_analyze[i])
        ar_changes.append(abs(ar_curr - ar_prev))
    centers = [((b[0]+b[2])//2, (b[1]+b[3])//2) for b in boxes_to_analyze]
    angle_changes, speed_changes = [], []
    for i in range(1, len(centers)):
        vec_curr = (centers[i][0] - centers[i-1][0], centers[i][1] - centers[i-1][1])
        vec_prev = vec_curr
        if i > 1: vec_prev = (centers[i-1][0] - centers[i-2][0], centers[i-1][1] - centers[i-2][1])
        
        # Gunakan MIN_MOVEMENT dari params
        if math.hypot(*vec_curr) > params['MIN_MOVEMENT'] and math.hypot(*vec_prev) > params['MIN_MOVEMENT']:
            dot = vec_curr[0]*vec_prev[0] + vec_curr[1]*vec_prev[1]
            mag_prod = math.hypot(*vec_curr) * math.hypot(*vec_prev)
            angle = 0 if mag_prod == 0 else math.degrees(math.acos(min(max(dot/mag_prod, -1.0), 1.0)))
            angle_changes.append(angle)
            speed_changes.append(abs(math.hypot(*vec_curr) - math.hypot(*vec_prev)))
            
    avg_ar = np.mean(ar_changes) if ar_changes else 0
    avg_angle = np.mean(angle_changes) if angle_changes else 0
    avg_speed = np.mean(speed_changes) if speed_changes else 0
    
    # Gunakan threshold dari params
    return any([
        avg_ar > params['AR_CHANGE_THRESH'] and (not ar_changes or max(ar_changes) > 2.0), # >2.0 adalah contoh, bisa juga bagian dari params
        avg_angle > params['ANGLE_CHANGE_THRESH'] and (not angle_changes or max(angle_changes) > 80), # >80 juga
        avg_speed > params['SPEED_CHANGE_THRESH'] and (not speed_changes or max(speed_changes) > 20), # >20 juga
        any(sc > 25 for sc in speed_changes)]) # 25 juga

# --- Fungsi Generator Frame untuk Streaming ---
# Modifikasi: generate_frames sekarang menerima `params`
def generate_frames(video_path, params):
    global model

    if model is None:
        # ... (error handling model tidak termuat, sama seperti sebelumnya) ...
        error_frame = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Error: Model YOLO Gagal Dimuat", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        while True:
            ret_flag, encoded = cv2.imencode(".jpg", error_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if not ret_flag: continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n')
            cv2.waitKey(1000)


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # ... (error handling video tidak bisa dibuka, sama seperti sebelumnya) ...
        error_frame = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Error: Video {video_path.split('/')[-1]} Gagal Dibuka", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        while True:
            ret_flag, encoded = cv2.imencode(".jpg", error_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if not ret_flag: continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n')
            cv2.waitKey(1000)
        return

    original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if original_frame_width > 0:
        processing_height = int(PROCESSING_WIDTH * (original_frame_height / original_frame_width))
    else:
        processing_height = int(PROCESSING_WIDTH * (9/16))

    middle_line_y = int(processing_height * 2/5)

    vehicle_count, accident_count = 0, 0                 
    counted_ids, counted_accidents = set(), set()          
    track_history = defaultdict(lambda: deque(maxlen=30))
    track_frames = defaultdict(int)     
    # Gunakan SMOOTHING_HISTORY dari params
    prev_boxes = defaultdict(lambda: deque(maxlen=params['SMOOTHING_HISTORY']))  
    abnormal_count = defaultdict(int)   
    accident_ids = set()                
    missed_frames_tracker = defaultdict(int)
    cooldown_frames = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            vehicle_count, accident_count = 0,0
            counted_ids.clear(); counted_accidents.clear()
            track_history.clear(); track_frames.clear(); prev_boxes.clear()
            abnormal_count.clear(); accident_ids.clear(); missed_frames_tracker.clear()
            cooldown_frames.clear()
            continue

        if frame.shape[1] != PROCESSING_WIDTH or frame.shape[0] != processing_height:
            frame_to_process = cv2.resize(frame, (PROCESSING_WIDTH, processing_height), interpolation=cv2.INTER_LINEAR)
        else:
            frame_to_process = frame.copy()
        
        current_processing_frame = frame_to_process
        
        results = model.track(
            current_processing_frame, persist=True, 
            conf=params['MIN_CONFIDENCE'], # Gunakan MIN_CONFIDENCE dari params
            iou=0.5, # IoU untuk NMS, bisa juga dimasukkan ke params jika perlu
            classes=VEHICLE_CLASSES, verbose=False, tracker="botsort.yaml"
        )

        current_frame_tracked_ids = set()
        current_collisions_this_frame = set()

        if results[0].boxes.id is not None:
            boxes_data = results[0].boxes.xyxy.cpu().numpy()
            ids_data = results[0].boxes.id.int().cpu().tolist()
            clss_data = results[0].boxes.cls.cpu().tolist()
            confs_data = results[0].boxes.conf.cpu().tolist()

            for i in range(len(ids_data)):
                for j in range(i + 1, len(ids_data)):
                    # Gunakan COLLISION_IOU_THRESH dari params
                    if calculate_iou(boxes_data[i], boxes_data[j]) > params['COLLISION_IOU_THRESH']:
                        current_collisions_this_frame.update({ids_data[i], ids_data[j]})

            for box, tid, cls, conf in zip(boxes_data, ids_data, clss_data, confs_data):
                current_frame_tracked_ids.add(tid)
                missed_frames_tracker[tid] = 0
                prev_boxes[tid].append(box)
                smoothed_box = smooth_boxes(box, prev_boxes[tid]) # Smoothing history diatur saat init prev_boxes
                x1, y1, x2, y2 = map(int, smoothed_box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                track_history[tid].append((cx, cy))
                track_frames[tid] += 1

                # Gunakan MIN_TRACK_FRAMES dari params
                if track_frames[tid] < params['MIN_TRACK_FRAMES']: continue

                is_abnormal = False
                if tid in current_collisions_this_frame:
                    is_abnormal = True; abnormal_count[tid] += 2  
                else:
                    # Kirim params ke check_abnormal_movement
                    is_abnormal = check_abnormal_movement(tid, smoothed_box, prev_boxes[tid], params)
                    if is_abnormal: abnormal_count[tid] += 1
                    else: abnormal_count[tid] = max(0, abnormal_count[tid] - 1)
                
                # Gunakan ACCIDENT_THRESHOLD dari params
                if abnormal_count[tid] >= params['ACCIDENT_THRESHOLD']:
                    if tid not in counted_accidents:
                        accident_count += 1; counted_accidents.add(tid)
                    accident_ids.add(tid); cooldown_frames[tid] = 0
                elif tid in accident_ids and abnormal_count[tid] < params['ACCIDENT_THRESHOLD'] // 2:
                    cooldown_frames[tid] += 1
                    if cooldown_frames[tid] > 30 and tid in accident_ids: accident_ids.remove(tid)

                color, status = BOX_COLOR, ""
                if tid in accident_ids: color, status = ACCIDENT_COLOR, "ACCIDENT!"
                elif tid in current_collisions_this_frame: color, status = WARNING_COLOR, "COLLISION!"
                elif is_abnormal: color, status = WARNING_COLOR, "WARNING!"
                
                cv2.rectangle(current_processing_frame, (x1,y1),(x2,y2),color,2)
                lbl = f"ID:{tid} {status}{model.names[int(cls)]} {conf:.2f}" # conf dari yolo, bukan params['MIN_CONFIDENCE']
                (lw,lh),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.4,1)
                cv2.rectangle(current_processing_frame,(x1,y1-lh-5),(x1+lw,y1),(0,0,0),-1)
                cv2.putText(current_processing_frame,lbl,(x1,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
                
                pts = np.array(track_history[tid],dtype=np.int32).reshape((-1,1,2))
                if len(pts) > 1: cv2.polylines(current_processing_frame,[pts],False,(255,0,0),1)

                if len(track_history[tid]) >= 2 and tid not in counted_ids:
                    if (track_history[tid][-2][1] < middle_line_y <= cy) or \
                       (track_history[tid][-2][1] > middle_line_y >= cy):
                        vehicle_count += 1; counted_ids.add(tid)

        ids_to_remove = []
        for tid_tr in list(track_history.keys()):
            if tid_tr not in current_frame_tracked_ids:
                missed_frames_tracker[tid_tr] += 1
                # Gunakan MAX_MISSED_FRAMES dari params
                if missed_frames_tracker[tid_tr] > params['MAX_MISSED_FRAMES']: ids_to_remove.append(tid_tr)
            elif 0 < missed_frames_tracker[tid_tr] <= params['MAX_MISSED_FRAMES']:
                 if prev_boxes[tid_tr]:
                    lx1,ly1,lx2,ly2 = map(int, prev_boxes[tid_tr][-1])
                    cv2.rectangle(current_processing_frame,(lx1,ly1),(lx2,ly2),(192,192,192),1)
        for tid_rem in ids_to_remove:
            for d_dict in [track_history,prev_boxes,abnormal_count,track_frames,missed_frames_tracker,cooldown_frames]:
                d_dict.pop(tid_rem,None)
            if tid_rem in accident_ids: accident_ids.remove(tid_rem)

        cv2.line(current_processing_frame, (0, middle_line_y), (PROCESSING_WIDTH, middle_line_y), LINE_COLOR, 1)
        cv2.putText(current_processing_frame, f"Kendaraan: {vehicle_count}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
        cv2.putText(current_processing_frame, f"Insiden: {accident_count}", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        (flag, encodedImage) = cv2.imencode(".jpg", current_processing_frame, encode_param)
        if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
    cap.release()

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('dashboard_page.html') 

@app.route('/video_feed_crash')
def video_feed_crash():
    # Gunakan parameter spesifik untuk 'crash.mp4'
    return Response(generate_frames(video_path='crash.mp4', params=VIDEO_SPECIFIC_PARAMS['crash']), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_highway')
def video_feed_highway():
    # Gunakan parameter spesifik untuk 'highway.mp4'
    return Response(generate_frames(video_path='highway.mp4', params=VIDEO_SPECIFIC_PARAMS['highway']), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print(f"Memulai Flask server dengan model: {YOLO_MODEL_PATH}, lebar proses: {PROCESSING_WIDTH}px, kualitas JPEG: {JPEG_QUALITY}")
    print("Parameter 'crash': ", VIDEO_SPECIFIC_PARAMS['crash'])
    print("Parameter 'highway': ", VIDEO_SPECIFIC_PARAMS['highway'])
    print("Akses di http://localhost:5000 atau http://<IP_LOKAL_ANDA>:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)