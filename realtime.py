import cv2
import numpy as np
from ultralytics import YOLO


face_model = YOLO("./yolov8n-face.pt")



def blur_face(face_roi, k=31):
    """Apply Gaussian Blur"""
    return cv2.GaussianBlur(face_roi, (k, k), 30)


def pixelate_face(face_roi, blocks=20):
    """Apply Pixelation"""
    h, w = face_roi.shape[:2]
    temp = cv2.resize(face_roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


def silhouette_face(face_roi, color=(0, 0, 0)):
    """Replace face with solid rectangle"""
    return np.full(face_roi.shape, color, dtype=np.uint8)


cap = cv2.VideoCapture(0)

method = "blur"  # change to "pixelate" or "silhouette"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run face detection
    results = face_model(frame, conf=0.4)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2

    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        face_roi = frame[y1:y2, x1:x2]

        if face_roi.size == 0:
            continue

        # Apply chosen anonymization
        if method == "blur":
            anon_face = blur_face(face_roi, k=31)
        elif method == "pixelate":
            anon_face = pixelate_face(face_roi, blocks=20)
        else:
            anon_face = silhouette_face(face_roi)

        frame[y1:y2, x1:x2] = anon_face

    # Show video
    cv2.imshow("Privacy Layer Demo", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("b"):
        method = "blur"
    elif key == ord("p"):
        method = "pixelate"
    elif key == ord("s"):
        method = "silhouette"

cap.release()
cv2.destroyAllWindows()
