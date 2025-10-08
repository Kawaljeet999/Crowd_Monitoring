import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load YOLO Face Detector

face_model = YOLO("./yolov8n-face.pt") # use yolov8 too


# 2. Anonymization methods

def blur_face(face_roi, k=31):

    return cv2.GaussianBlur(face_roi, (k, k), 30)

def pixelate_face(face_roi, blocks=20):

    h, w = face_roi.shape[:2]
    temp = cv2.resize(face_roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def silhouette_face(face_roi, color=(0, 0, 0)):

    return np.full(face_roi.shape, color, dtype=np.uint8)


frame = cv2.imread("./sample.jpg")

if frame is None:
    print("Error: Image not found!")
    exit()

method = "blur"  # change to "pixelate" or "silhouette"

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

cv2.imshow("Anonymized Image", frame)
cv2.imwrite("anonymized_output(yolov8n-face).jpg", frame)  # <- Save output image
print("Saved anonymized image as anonymized_output.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()
