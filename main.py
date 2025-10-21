import cv2
from ultralytics import YOLO
from collections import deque, Counter

VIDEO = r"C:\Users\HH\OneDrive\Desktop\car_test.mp4"
MODEL = r"C:\Users\84827\ParkingBTN\best1.pt"

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)
gate, spots, sid, fc, brightness, contrast = None, {}, 0, 0, 50, 50

dist = lambda p1, p2: ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5
adjust = lambda img, b, c: cv2.convertScaleAbs(img, alpha=c/50, beta=(b-50)*2)

def valid(box, conf):
    x1, y1, x2, y2 = box
    area, ratio = (x2-x1)*(y2-y1), max(x2-x1,y2-y1)/(min(x2-x1,y2-y1)+1)
    return 1000 < area < 32000 and ratio < 5 and conf > 0.4

def smooth(hist):
    if len(hist) < 10: return Counter(hist).most_common(1)[0][0]
    recent = list(hist)[-10:]
    free = sum(1 for x in recent if x == 'free')
    return 'car' if len(recent)-free >= 7 else 'free' if free >= 7 else Counter(hist).most_common(1)[0][0]

def detect(frm):
    global spots, sid
    for s in spots: spots[s]['age'] += 1
    for r in model(frm, conf=0.42, iou=0.4, verbose=False):
        for box in r.boxes:
            lbl = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            if not valid((x1,y1,x2,y2), float(box.conf[0])): continue
            cx, cy = (x1+x2)//2, (y1+y2)//2
            match = min(spots.items(), key=lambda s: dist((cx,cy), s[1]['c']), default=(None,{'c':(9999,9999)}))[0] if spots else None
            if match and dist((cx,cy), spots[match]['c']) < 60:
                o = spots[match]
                if 'h' not in o: o['h'] = deque([o['l']], maxlen=15)
                o['h'].append(lbl)
                spots[match] = {'c':(int(0.7*cx+0.3*o['c'][0]),int(0.7*cy+0.3*o['c'][1])),
                                'b':(x1,y1,x2,y2),'l':smooth(o['h']),'age':0,'n':o['n']+1,'h':o['h']}
            else:
                sid += 1
                spots[sid] = {'c':(cx,cy),'b':(x1,y1,x2,y2),'l':lbl,'age':0,'n':1,'h':deque([lbl],maxlen=15)}
    spots = {s:d for s,d in spots.items() if d['age'] < 200}

print("\n🚗 HỆ THỐNG QUẢN LÝ BÃI ĐỖ XE\n" + "="*60)
print("📍 Click chọn cổng | ⌨️ 'q' thoát | 'r' reset\n")

cv2.namedWindow("Parking")
cv2.setMouseCallback("Parking", lambda e,x,y,f,p: globals().update(gate=(x,y)) if e==1 else None)
cv2.createTrackbar("Brightness", "Parking", 50, 100, lambda v: globals().update(brightness=v))
cv2.createTrackbar("Contrast", "Parking", 50, 100, lambda v: globals().update(contrast=v))

while True:
    ret, frm = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0); spots, sid, fc = {}, 0, 0; continue
    frm = cv2.resize(adjust(frm, brightness, contrast), None, fx=0.7, fy=0.7)
    fc += 1; h, w = frm.shape[:2]
    if fc % 2 == 0: detect(frm)

    stable = {s:d for s,d in spots.items() if d['n'] >= 2}
    frees = {s:d for s,d in stable.items() if d['l']=='free'}
    total, empty = len(stable), len(frees)
    closest = min(frees.values(), key=lambda s: dist(gate,s['c']), default=None) if gate and frees else None

    for d in stable.values():
        x1,y1,x2,y2 = d['b']
        cv2.rectangle(frm,(x1,y1),(x2,y2),(0,0,255) if d['l']=='car' else (0,255,0),2)
    if gate:
        cv2.circle(frm,gate,20,(255,255,0),-1)
        cv2.putText(frm,"CONG",(gate[0]-35,gate[1]-30),0,0.8,(255,255,0),2)
        if closest:
            cv2.line(frm,gate,closest['c'],(0,255,255),4)
            cv2.circle(frm,closest['c'],45,(0,255,255),5)
            mid=((gate[0]+closest['c'][0])//2,(gate[1]+closest['c'][1])//2)
            cv2.circle(frm,mid,32,(0,0,0),-1)
            cv2.putText(frm,f"{int(dist(gate,closest['c']))}",(mid[0]-24,mid[1]+10),0,0.7,(0,255,255),2)
    else:
        cv2.putText(frm,">>> CLICK CONG <<<",(w//2-200,h//2),0,1,(0,255,255),3)

    cv2.rectangle(frm,(10,10),(460,165),(0,0,0),-1)
    cv2.rectangle(frm,(10,10),(460,165),(255,255,255),3)
    cv2.putText(frm,"BAI DAU XE",(25,50),0,1,(255,255,255),3)
    cv2.putText(frm,f"Tong: {total}",(25,85),0,0.75,(200,200,200),2)
    cv2.putText(frm,f"Trong: {empty}",(25,115),0,0.85,(0,255,0),2)
    cv2.putText(frm,f"Day: {total-empty}",(250,115),0,0.85,(0,0,255),2)
    if gate and empty==0 and total>0:
        cv2.rectangle(frm,(10,125),(460,158),(0,0,255),-1)
        cv2.putText(frm,"!!! HET CHO !!!",(40,148),0,0.8,(255,255,255),3)
    elif gate and closest:
        cv2.putText(frm,f"Gan: {int(dist(gate,closest['c']))}px",(25,148),0,0.65,(0,255,255),2)

    cv2.imshow("Parking", frm)
    key = cv2.waitKey(30)&0xFF
    if key==ord('q'): break
    elif key==ord('r'): gate=None

cap.release()
cv2.destroyAllWindows()