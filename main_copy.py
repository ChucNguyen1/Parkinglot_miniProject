# from ultralytics import YOLO
# import cv2

# # Load model
# model = YOLO(r"D:\PycharmProjects\ParkingBTN\best.pt")

# # Mở video
# cap = cv2.VideoCapture(r"D:\PycharmProjects\ParkingBTN\samples\parking_1920_1080_loop.mp4")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Dự đoán trên từng frame
#     results = model(frame, conf=0.5)
#     for r in results:
#         img_array = r.plot()  # Vẽ bounding box lên frame

#         cv2.imshow('Parking Detect', img_array)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# from ultralytics import YOLO

# # Load YOLO model đã train
# model = YOLO("bestt.pt")  # thay bằng đường dẫn model của bạn

# # video_path = './samples/parking_1920_1080_loop.mp4'
# video_path ="D:\PycharmProjects\ParkingBTN\samples\parking_1920_1080_loop.mp4"
# cap = cv2.VideoCapture(video_path)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Chạy detection
#     results = model(frame)

#     # Vẽ kết quả
#     for r in results:
#         boxes = r.boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2]
#         classes = r.boxes.cls.cpu().numpy()  # class id
#         confs = r.boxes.conf.cpu().numpy()   # confidence

#         for box, cls, conf in zip(boxes, classes, confs):
#             x1, y1, x2, y2 = map(int, box)
#             if cls == 0:  # giả sử class 0 = empty
#                 color = (0, 255, 0)
#             else:         # class 1 = no_empty
#                 color = (0, 0, 255)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#     cv2.imshow("Parking Detect", frame)
#     if cv2.waitKey(25) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
# import cv2
# from ultralytics import YOLO

# # Load YOLO model đã train
# model = YOLO("bestt.pt")  # thay bằng đường dẫn model của bạn

# # Đọc ảnh
# image_path = r"D:\PycharmProjects\ParkingBTN\data\test\images\1-7-_jpg.rf.5a746ea3d5baf4b8f7b9d3cfb36c2d31.jpg"
# frame = cv2.imread(image_path)

# # Chạy detection
# results = model(frame, conf=0.5)

# # Vẽ kết quả
# for r in results:
#     boxes = r.boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2]
#     classes = r.boxes.cls.cpu().numpy()  # class id
#     confs = r.boxes.conf.cpu().numpy()   # confidence

#     for box, cls, conf in zip(boxes, classes, confs):
#         x1, y1, x2, y2 = map(int, box)
#         if cls == 0:  # giả sử class 0 = empty
#             color = (0, 255, 0)  # xanh
#             label = f"Empty {conf:.2f}"
#         else:         # class 1 = no_empty
#             color = (0, 0, 255)  # đỏ
#             label = f"Occupied {conf:.2f}"

#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, color, 2)

# # Hiển thị ảnh kết quả
# cv2.imshow("Parking Detect", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# from ultralytics import YOLO

# # Load mô hình YOLOv8 đã train
# model = YOLO(r"D:\PycharmProjects\ParkingBTN\bestt.pt")   # đổi đường dẫn tới file best.pt của bạn

# # Mở video
# cap = cv2.VideoCapture(r"D:\PycharmProjects\ParkingBTN\samples\parking_1920_1080_loop.mp4")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)

#     empty_count = 0
#     total_spots = 0

#     for r in results:
#         for box in r.boxes:
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

#             label = model.names[cls_id]  # "bos" hoặc "dolu"
#             color = (0, 255, 0) if label == "bos" else (0, 0, 255)

#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#             total_spots += 1
#             if label == "bos":
#                 empty_count += 1

#     # Hiển thị số chỗ trống
#     cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
#     cv2.putText(frame, f"Available spots: {empty_count} / {total_spots}",
#                 (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#     cv2.imshow("Parking Detection", frame)
#     if cv2.waitKey(25) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2 
from ultralytics import YOLO 

# Load mô hình YOLOv8 đã train
model = YOLO(r"C:\Users\84827\ParkingBTN\bestt.pt")   # đổi đường dẫn tới file best.pt của bạn

# Đọc ảnh
image_path = r"C:\Users\84827\Pictures\I'm oke\parking-lot-4928423.webp"  # đổi sang ảnh của bạn
frame = cv2.imread(image_path)

# Chạy YOLOv8 trên ảnh
results = model(frame)

empty_count = 0
total_spots = 0

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        label = model.names[cls_id]  # "bos" hoặc "dolu"
        color = (0, 255, 0) if label == "empty" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        total_spots += 1
        if label == "empty":
            empty_count += 1

# Hiển thị số chỗ trống
cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
cv2.putText(frame, f"Available spots: {empty_count} / {total_spots}",
            (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Hiển thị ảnh kết quả
cv2.imshow("Parking Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# yolo task=detect mode=train model=yolov8s.pt data="D:/PycharmProjects/ParkingBTN/haha/data.yaml" epochs=100 imgsz=640 batch=8 augment=True degrees=10 scale=0.5 flipud=0.5 hsv_h=0.05 hsv_s=0.5 hsv_v=0.5 device=0

# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image

# # Load YOLOv8 model (đổi đường dẫn nếu cần)
# model = YOLO(r"D:\PycharmProjects\ParkingBTN\best.pt")

# st.title("Parking Slot Detection Demo")
# st.write("Upload a parking lot image to detect empty and occupied spots.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Đọc ảnh từ upload
#     image = Image.open(uploaded_file).convert("RGB")
#     frame = np.array(image)

#     # Chạy YOLOv8
#     results = model(frame)

#     empty_count = 0
#     total_spots = 0

#     for r in results:
#         for box in r.boxes:
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

#             label = model.names[cls_id]  # "bos" hoặc "dolu" hoặc tên class bạn đặt
#             color = (0, 255, 0) if label == "bos" else (0, 0, 255)

#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#             total_spots += 1
#             if label == "bos":
#                 empty_count += 1

#     # Hiển thị số chỗ trống
#     cv2.rectangle(frame, (10, 10), (400, 60), (0, 0, 0), -1)
#     cv2.putText(frame, f"Available spots: {empty_count} / {total_spots}",
#                 (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#     st.image(frame, channels="RGB", caption="Detection Result")