import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from models.utils.face_detection import MultiFaceDetectorCNN
import time



class MultiFaceDetector:
    def __init__(self, model_path=None, confidence_threshold=0.5, iou_threshold=0.4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MultiFaceDetectorCNN().to(self.device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        
        self.model.eval()
    
    def detect_faces(self, image, confidence_threshold=0.5):
        """Detect multiple faces in an image"""
        original_h, original_w = image.shape[:2]
        
        # Preprocess image
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (512, 512))
        input_img = input_img.transpose(2, 0, 1)
        input_img = torch.tensor(input_img, dtype=torch.float32).unsqueeze(0) / 255.0
        input_img = input_img.to(self.device)
        
        with torch.no_grad():
            pred_bboxes, pred_faces = self.model(input_img)
        
        pred_bboxes = pred_bboxes[0].cpu().numpy()
        pred_faces = pred_faces[0].cpu().numpy()
        
        detected_faces = []
        
        for i in range(len(pred_faces)):
            confidence = pred_faces[i]
            if confidence > confidence_threshold:
                x, y, w, h = pred_bboxes[i]
                
                # Scale back to original image coordinates
                x = int(x * original_w)
                y = int(y * original_h)
                w = int(w * original_w)
                h = int(h * original_h)
                
                # Apply non-maximum suppression
                valid = True
                for j, (existing_x, existing_y, existing_w, existing_h, existing_conf) in enumerate(detected_faces):
                    iou = self._calculate_iou((x, y, w, h), (existing_x, existing_y, existing_w, existing_h))
                    if iou > self.iou_threshold:
                        if confidence > existing_conf:
                            detected_faces[j] = (x, y, w, h, confidence)
                        valid = False
                        break
                
                if valid:
                    detected_faces.append((x, y, w, h, confidence))
        
        return detected_faces
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to [x1, y1, x2, y2] format
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
        
        # Calculate intersection area
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def process_video(self, video_path=0, output_path=None):
        """Process video for crowd analysis"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer if output path is specified
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Processing video for crowd analysis...")
        print("Press 'q' to quit")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = self.detect_faces(frame, self.confidence_threshold)
            
            # Draw results
            for (x, y, w, h, confidence) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'{confidence:.2f}', 
                           (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 1)
            
            # Display crowd count
            cv2.putText(frame, f'People: {len(faces)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2)
            
            if output_path:
                out.write(frame)
            
            cv2.imshow('Crowd Analysis', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        end_time = time.time()
        print(f"Processed {frame_count} frames in {end_time - start_time:.2f} seconds")
        print(f"Average FPS: {frame_count / (end_time - start_time):.2f}")
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()