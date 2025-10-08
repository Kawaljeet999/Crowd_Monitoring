import cv2
import mediapipe as mp
import numpy as np


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)


def blur_face(face_roi, k=31):
    """Apply Gaussian Blur"""
    return cv2.GaussianBlur(face_roi, (k, k), 30)

def pixelate_face(face_roi, blocks=20):
    """Apply Pixelation"""
    h, w = face_roi.shape[:2]
    temp = cv2.resize(face_roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def silhouette_face(face_roi, color=(0, 0, 0)):
    """Replace face with solid color"""
    return np.full(face_roi.shape, color, dtype=np.uint8)


source = 0
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("❌ Error: Could not open video source.")
    exit()

method = "blur"
print("🔹 Press 'b' = blur | 'p' = pixelate | 's' = silhouette | 'q' = quit")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB (MediaPipe requires this)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            # Convert relative coords to pixel values
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Extract the face region
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            # Apply anonymization
            if method == "blur":
                anon_face = blur_face(face_roi)
            elif method == "pixelate":
                anon_face = pixelate_face(face_roi)
            else:
                anon_face = silhouette_face(face_roi)

            # Replace the original face region
            frame[y1:y2, x1:x2] = anon_face

    # Show the anonymized video
    cv2.imshow("MediaPipe Face Anonymizer", frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("b"):
        method = "blur"
        print("🌀 Switched to: BLUR")
    elif key == ord("p"):
        method = "pixelate"
        print("🟦 Switched to: PIXELATE")
    elif key == ord("s"):
        method = "silhouette"
        print("⬛ Switched to: SILHOUETTE")

cap.release()
cv2.destroyAllWindows()
