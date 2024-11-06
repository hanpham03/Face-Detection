import sys
import os
import cv2
import mediapipe as mp
import face_recognition
import numpy as np

# Thay đổi đường dẫn đến thư mục chứa face_recognition_models
sys.path.append(os.path.join(os.path.dirname(__file__), 'face_recognition_models'))

# Tải dữ liệu khuôn mặt đã biết
known_face_encodings = []
known_face_names = []

# Đường dẫn đến thư mục chứa ảnh khuôn mặt đã biết
path_to_images = os.path.join(os.path.dirname(__file__), 'face_images')

# Duyệt qua từng ảnh trong thư mục và lưu mã hóa
for filename in os.listdir(path_to_images):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(path_to_images, filename)
        image = face_recognition.load_image_file(image_path)

        # Kiểm tra mã hóa khuôn mặt
        encodings = face_recognition.face_encodings(image)
        if encodings:  # Nếu có ít nhất một khuôn mặt được tìm thấy
            encoding = encodings[0]
            known_face_encodings.append(encoding)

            # Lấy tên từ tên file
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
        else:
            print(f"No face found in {filename}")

# Khởi tạo Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Mở video từ webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi khung hình BGR sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    # Nếu có phát hiện khuôn mặt
    if results.detections:
        for detection in results.detections:
            # Chuyển bounding box từ tỷ lệ sang pixel
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            bbox = (
                int(bboxC.xmin * w),
                int(bboxC.ymin * h),
                int(bboxC.width * w),
                int(bboxC.height * h)
            )

            # Trích xuất khuôn mặt
            face_frame = frame_rgb[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

            # Mã hóa khuôn mặt và so sánh với khuôn mặt đã biết
            face_encodings = face_recognition.face_encodings(face_frame)
            if face_encodings:
                face_encoding = face_encodings[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Tìm khoảng cách phù hợp nhất
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                # Hiển thị tên lên khuôn mặt
                cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 2)
                mp_drawing.draw_detection(frame, detection)

    # Hiển thị khung hình với nhận diện khuôn mặt
    cv2.imshow('Nhan dien khuon mat bang python', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng tất cả các cửa sổ
cap.release()
cv2.destroyAllWindows()
