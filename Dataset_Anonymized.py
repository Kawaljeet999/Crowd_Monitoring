import cv2
from ultralytics import YOLO
import mediapipe as mp

# --- Load Models ---
yolo_face = YOLO("yolov8n-face.pt")
yolo_person = YOLO("yolov8n.pt")
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# --- Ground Truth --- (example: manually labeled faces)
# Format: list of bounding boxes [x1, y1, x2, y2] per image/frame
ground_truth = [
    # For example frame 0
    [[100, 50, 160, 110], [300, 70, 360, 130]],
    # For example frame 1
    [[90, 40, 150, 100]]
]


# --- Function to compute IOU ---
def iou(boxA, boxB):
    # box = [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# --- Evaluate a method ---
def evaluate_yolo_face(image, gt_boxes, conf=0.4):
    results = yolo_face(image, conf=conf)
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()
    TP = 0
    for gt in gt_boxes:
        for pred in pred_boxes:
            if iou(gt, pred) > 0.5:
                TP += 1
                break
    return TP, len(gt_boxes)


# --- Main Loop (Example with images) ---
total_TP = 0
total_gt = 0
image_files = ["sample.jpg"]

for i, img_path in enumerate(image_files):
    img = cv2.imread(img_path)
    TP, gt_count = evaluate_yolo_face(img, ground_truth[i])
    total_TP += TP
    total_gt += gt_count

accuracy = total_TP / total_gt
print("Detection Accuracy:", accuracy)
