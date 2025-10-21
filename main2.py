import cv2
from ultralytics import YOLO
import numpy as np

VIDEO = r"C:\Users\84827\ParkingBTN\Video\main2\carPark.mp4"
MODEL = r"C:\Users\84827\ParkingBTN\best1.pt"
model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)
gate = None
brightness, contrast = 50, 50
adjust = lambda img, b, c: cv2.convertScaleAbs(img, alpha=c/50, beta=(b-50)*2)

def iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = (x2_1-x1_1)*(y2_1-y1_1) + (x2_2-x1_2)*(y2_2-y1_2) - inter
    return inter / union if union > 0 else 0

def merge_overlapping_boxes(boxes, iou_threshold=0.3):
    if not boxes: return []
    merged, used = [], [False] * len(boxes)
    for i, box1 in enumerate(boxes):
        if used[i]: continue
        cluster = [box1]
        used[i] = True
        for j, box2 in enumerate(boxes):
            if i != j and not used[j] and iou(box1['box'], box2['box']) > iou_threshold:
                cluster.append(box2)
                used[j] = True
        if cluster:
            largest = max(cluster, key=lambda b: (b['box'][2]-b['box'][0])*(b['box'][3]-b['box'][1]))
            merged.append({'box': largest['box'], 'label': max(set([b['label'] for b in cluster]), key=[b['label'] for b in cluster].count)})
    return merged

def distance_point_to_segment(point, seg_start, seg_end):
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0: return np.sqrt((px - x1)**2 + (py - y1)**2)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return np.sqrt((px-(x1+t*dx))**2 + (py-(y1+t*dy))**2)

def path_blocked_by_box(start, end, box, buffer=40):
    x1, y1, x2, y2 = box
    return distance_point_to_segment(((x1+x2)//2, (y1+y2)//2), start, end) < buffer

def find_accessible_spot(gate, free_spots, obstacles):
    if not gate or not free_spots: return None
    candidates = []
    for spot in free_spots:
        x1, y1, x2, y2 = spot['box']
        spot_center = ((x1+x2)//2, (y1+y2)//2)
        euclidean_dist = np.sqrt((gate[0]-spot_center[0])**2 + (gate[1]-spot_center[1])**2)
        blocking_count = sum(1 for obs in obstacles if path_blocked_by_box(gate, spot_center, obs['box'], buffer=40))
        candidates.append({'spot': spot, 'center': spot_center, 'distance': euclidean_dist, 'blocked_count': blocking_count})
    candidates.sort(key=lambda x: (x['blocked_count'], x['distance']))
    return candidates[0] if candidates else None

print("\n🚗 BÃI ĐỖ XE | 📍 Click cổng | ⌨️ 'q' thoát\n")
cv2.namedWindow("Parking")
cv2.setMouseCallback("Parking", lambda e,x,y,f,p: globals().update(gate=(x,y)) if e==1 else None)
cv2.createTrackbar("Brightness", "Parking", 50, 100, lambda v: globals().update(brightness=v))
cv2.createTrackbar("Contrast", "Parking", 50, 100, lambda v: globals().update(contrast=v))

while True:
    ret, frm = cap.read()
    if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
    frm = cv2.resize(adjust(frm, brightness, contrast), None, fx=0.7, fy=0.7)
    h, w = frm.shape[:2]
    all_boxes = []
    for r in model(frm, conf=0.42, iou=0.3, verbose=False):
        for box in r.boxes:
            lbl = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            area = (x2-x1) * (y2-y1)
            ratio = max(x2-x1, y2-y1) / (min(x2-x1, y2-y1) + 1)
            if 800 < area < 26000 and ratio < 6.0 and conf > 0.42:
                all_boxes.append({'box': (x1, y1, x2, y2), 'label': lbl})
    merged_boxes = merge_overlapping_boxes(all_boxes, iou_threshold=0.3)
    boxes_car = [b for b in merged_boxes if b['label'] == 'car']
    boxes_free = [b for b in merged_boxes if b['label'] == 'free']
    total, empty = len(merged_boxes), len(boxes_free)
    closest_result = find_accessible_spot(gate, boxes_free, boxes_car)
    for b in boxes_car: cv2.rectangle(frm, b['box'][:2], b['box'][2:], (0,0,255), 2)
    for b in boxes_free: cv2.rectangle(frm, b['box'][:2], b['box'][2:], (0,255,0), 2)
    if gate:
        cv2.circle(frm, gate, 20, (255,255,0), -1)
        cv2.circle(frm, gate, 23, (0,0,0), 3)
        cv2.putText(frm, "CONG", (gate[0]-35, gate[1]-30), 0, 0.8, (255,255,0), 2)
        if closest_result:
            cv2.line(frm, gate, closest_result['center'], (0,255,255), 4)
            cv2.rectangle(frm, closest_result['spot']['box'][:2], closest_result['spot']['box'][2:], (0,255,255), 4)
            mid = ((gate[0]+closest_result['center'][0])//2, (gate[1]+closest_result['center'][1])//2)
            cv2.circle(frm, mid, 32, (0,0,0), -1)
            cv2.putText(frm, f"{int(closest_result['distance'])}", (mid[0]-24, mid[1]+10), 0, 0.7, (0,255,255), 2)
    else:
        cv2.putText(frm, ">>> CLICK CONG <<<", (w//2-200, h//2), 0, 1, (0,255,255), 3)
    cv2.rectangle(frm, (10,10), (300, 125), (0,0,0), -1)
    cv2.rectangle(frm, (10,10), (300, 125), (255,255,255), 2)
    cv2.putText(frm, "BAI DAU XE", (20,38), 0, 0.7, (255,255,255), 2)
    cv2.putText(frm, f"Tong: {total}", (20,65), 0, 0.6, (200,200,200), 1)
    cv2.putText(frm, f"Trong: {empty}", (20,88), 0, 0.65, (0,255,0), 2)
    cv2.putText(frm, f"Day: {total-empty}", (180,88), 0, 0.65, (0,0,255), 2)
    if gate and empty==0 and total>0:
        cv2.rectangle(frm, (10,95), (300, 120), (0,0,255), -1)
        cv2.putText(frm, "!!! HET CHO !!!", (30,112), 0, 0.6, (255,255,255), 2)
    elif gate and closest_result:
        cv2.putText(frm, f"Gan: {int(closest_result['distance'])}px", (20,112), 0, 0.55, (0,255,255), 1)
    cv2.imshow("Parking", frm)
    if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()