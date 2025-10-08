import cv2
import numpy as np
from ultralytics import YOLO


face_model = YOLO("./yolov8n-face.pt") #yolov8n <-- another model

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


video_path = "./videoplayback.mp4"  # <- Change to your input video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video info
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
output_path = "anonymized_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

method = "blur"  # Change to "pixelate" or "silhouette"


frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing {frame_count} frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
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

    out.write(frame)

    # Optional: Show progress live
    cv2.imshow("Anonymized Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Saved anonymized video as {output_path}")
