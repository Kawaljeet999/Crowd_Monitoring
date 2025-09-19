import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import scipy.io
from sklearn.model_selection import train_test_split
import time
import json
from models.utils.face_detection import MultiFaceDetectorCNN

class MultiFaceDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, img_size=512, max_faces=20):
        """
        Dataset class for multiple face detection with .mat annotations
        
        Args:
            image_dir: Directory containing images
            annotation_dir: Directory containing .mat annotation files
            transform: Optional transforms
            img_size: Target image size
            max_faces: Maximum number of faces to handle per image
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.transform = transform
        self.img_size = img_size
        self.max_faces = max_faces
        
        # Get all image files
        self.image_files = list(self.image_dir.glob('*.jpg')) + \
                          list(self.image_dir.glob('*.jpeg')) + \
                          list(self.image_dir.glob('*.png'))
        
        print(f"Found {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            return self.__getitem__((idx + 1) % len(self))
        
        original_h, original_w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotation from .mat file
        annotation_path = self.annotation_dir / (img_path.stem + '.mat')
        
        all_bboxes = []
        all_face_masks = []
        
        if annotation_path.exists():
            try:
                # Load .mat file
                mat_data = scipy.io.loadmat(str(annotation_path))
                
                # Extract multiple bounding boxes - adjust this based on your .mat structure
                # Common formats: multiple 'box' arrays or a single array with multiple rows
                if 'boxes' in mat_data and len(mat_data['boxes'].shape) > 1:
                    # Multiple boxes in 2D array
                    boxes = mat_data['boxes']
                    for i in range(min(boxes.shape[0], self.max_faces)):
                        bbox = boxes[i]  # [x, y, width, height]
                        # Normalize coordinates
                        normalized_bbox = [
                            bbox[0] / original_w, 
                            bbox[1] / original_h,
                            bbox[2] / original_w,
                            bbox[3] / original_h
                        ]
                        all_bboxes.append(normalized_bbox)
                        all_face_masks.append(1.0)
                
                elif 'box' in mat_data:
                    # Single box or multiple boxes in different format
                    boxes = mat_data['box']
                    if boxes.ndim == 1:
                        # Single face
                        normalized_bbox = [
                            boxes[0] / original_w, 
                            boxes[1] / original_h,
                            boxes[2] / original_w,
                            boxes[3] / original_h
                        ]
                        all_bboxes.append(normalized_bbox)
                        all_face_masks.append(1.0)
                    else:
                        # Multiple faces
                        for i in range(min(boxes.shape[0], self.max_faces)):
                            bbox = boxes[i]
                            normalized_bbox = [
                                bbox[0] / original_w, 
                                bbox[1] / original_h,
                                bbox[2] / original_w,
                                bbox[3] / original_h
                            ]
                            all_bboxes.append(normalized_bbox)
                            all_face_masks.append(1.0)
                
                else:
                    # Try to find any array with face coordinates
                    for key in mat_data:
                        if (isinstance(mat_data[key], np.ndarray) and 
                            mat_data[key].size >= 4 and 
                            mat_data[key].ndim >= 1):
                            boxes = mat_data[key]
                            if boxes.ndim == 1:
                                boxes = boxes.reshape(1, -1)
                            
                            for i in range(min(boxes.shape[0], self.max_faces)):
                                if boxes.shape[1] >= 4:
                                    bbox = boxes[i][:4]
                                    normalized_bbox = [
                                        bbox[0] / original_w, 
                                        bbox[1] / original_h,
                                        bbox[2] / original_w,
                                        bbox[3] / original_h
                                    ]
                                    all_bboxes.append(normalized_bbox)
                                    all_face_masks.append(1.0)
                            break
                
            except Exception as e:
                print(f"Error loading annotation {annotation_path}: {e}")
        
        # Pad to max_faces
        while len(all_bboxes) < self.max_faces:
            all_bboxes.append([0, 0, 0, 0])
            all_face_masks.append(0.0)
        
        # Convert to tensors
        bboxes_tensor = torch.tensor(all_bboxes[:self.max_faces], dtype=torch.float32)
        face_masks_tensor = torch.tensor(all_face_masks[:self.max_faces], dtype=torch.float32)
        
        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        return image, bboxes_tensor, face_masks_tensor